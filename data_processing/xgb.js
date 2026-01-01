// CẤU HÌNH MÔ HÌNH
var RESOLUTION = 30;
var TRAIN_SPLIT = 0.7;
var BUFFER_SIZE = 15;

// ========== PSO OPTIMIZATION (R²=0.7900, Fitness=0.4750) ==========
// var NUM_TREES = 1000;
// var LEARNING_RATE = 0.01;
// var SHRINKAGE = LEARNING_RATE;

// ========== PUMA OPTIMIZATION (R²=0.7913, Fitness=0.4793) - ACTIVE - BEST ==========
var NUM_TREES = 813;
var LEARNING_RATE = 0.01;
var SHRINKAGE = LEARNING_RATE;

// ========== RANDOMIZED SEARCH  ==========
// var NUM_TREES = 78;
// var LEARNING_RATE = 0.96;  
// var SHRINKAGE = LEARNING_RATE;

var featureNames = [
  'lulc', 'Density_River', 'Density_Road', 'Distan2river', 'Distan2road_met',
  'aspect', 'curvature', 'dem', 'flowDir', 'slope', 'twi', 'NDVI', 'rainfall'
];

var floodPoints = floodPoints.map(function(feature) {
  var lon = ee.Number(feature.get('lon'));
  var lat = ee.Number(feature.get('lat'));
  var floodValue = ee.Number(feature.get('flood'));
  return feature.set('flood', floodValue)
                .setGeometry(ee.Geometry.Point([lon, lat], 'EPSG:4326'));
});

var features = imageAsset.select(featureNames).clip(studyArea);

var floodPointsBuffered = floodPoints.map(function(feature) {
  var buffered = feature.buffer(BUFFER_SIZE);
  return buffered.copyProperties(feature, ['flood', 'lat', 'lon']);
});

var trainingData = features.sampleRegions({
  collection: floodPointsBuffered,
  properties: ['flood', 'lat', 'lon'],
  scale: 10,
  tileScale: 1,
  geometries: false
});

var requiredColumns = ['flood'].concat(featureNames);
trainingData = trainingData.filter(ee.Filter.notNull(requiredColumns));

var withRandom = trainingData.randomColumn('random', 42);
var training = withRandom.filter(ee.Filter.lt('random', TRAIN_SPLIT));
var validation = withRandom.filter(ee.Filter.gte('random', TRAIN_SPLIT));

var xgbRegressor = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: NUM_TREES,
  shrinkage: SHRINKAGE,
  maxNodes: null,
  loss: 'LeastAbsoluteDeviation',
  seed: 42
}).setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'flood',
    inputProperties: featureNames
  });

var floodsusceptibility = features.classify(xgbRegressor)
  .rename('flood_susceptibility')
  .clamp(0, 1)
  .reproject({
    crs: 'EPSG:4326',
    scale: RESOLUTION
  }).clip(studyArea);

// ===== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION =====
// Tạo lại geometry cho validation points
var validationWithGeom = validation.map(function(feature) {
  var lon = ee.Number(feature.get('lon'));
  var lat = ee.Number(feature.get('lat'));
  return feature.setGeometry(ee.Geometry.Point([lon, lat], 'EPSG:4326'));
});

// Classify toàn bộ ảnh features
var predictions = features.classify(xgbRegressor).rename('prediction');

// Sample predictions tại các điểm validation
var validationPredictions = predictions.sampleRegions({
  collection: validationWithGeom,
  properties: ['flood', 'lat', 'lon'],
  scale: RESOLUTION,
  tileScale: 1,
  geometries: false
});

print('XGB Validation Count:', validationPredictions.size());
print('XGB Validation Sample:', validationPredictions.limit(10));

// Tính R² score
var calculateR2 = function(collection) {
  var predicted = ee.Array(collection.aggregate_array('prediction'));
  var observed = ee.Array(collection.aggregate_array('flood'));
  
  var meanObserved = observed.reduce(ee.Reducer.mean(), [0]).get([0]);
  
  var ssRes = observed.subtract(predicted).pow(2).reduce(ee.Reducer.sum(), [0]).get([0]);
  var ssTot = observed.subtract(meanObserved).pow(2).reduce(ee.Reducer.sum(), [0]).get([0]);
  
  var r2 = ee.Number(1).subtract(ee.Number(ssRes).divide(ssTot));
  return r2;
};

var r2Score = calculateR2(validationPredictions);
print('XGB R² Score:', r2Score);

// Export validation data để vẽ 2D histogram
Export.table.toDrive({
  collection: validationPredictions.select(['flood', 'prediction', 'lat', 'lon']),
  description: 'validation_XGB',
  fileFormat: 'CSV'
});

// Hiển thị bản đồ
Map.centerObject(studyArea, 10);
Map.addLayer(floodsusceptibility, {
  min: 0,
  max: 1,
  palette: ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']
}, 'Flood susceptibility');

Export.image.toDrive({
  image: floodsusceptibility,
  description: 'flood_susceptibility_XGB',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  crs: 'EPSG:4326'
});
