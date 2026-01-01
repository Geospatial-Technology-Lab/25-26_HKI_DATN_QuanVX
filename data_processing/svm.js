// CẤU HÌNH MÔ HÌNH SVR
var RESOLUTION = 30;
var TRAIN_SPLIT = 0.7;
var BUFFER_SIZE = 15;

// ========== PSO OPTIMIZATION (R²=0.5923, Fitness=0.0991) - ACTIVE ==========
var SVM_KERNEL = 'RBF';
var SVM_C = 1.193;            // 1.1928841970033406
var SVM_GAMMA = 0.0787;       // 0.0787381332672518
var SVM_EPSILON = 0.118;      // 0.1177825297466713

// ========== PUMA OPTIMIZATION ==========
// var SVM_KERNEL = 'RBF';
// var SVM_C = 408.007;          // 408.0072353832169
// var SVM_GAMMA = 0.2225;       // 0.22246591790356623
// var SVM_EPSILON = 0.01;

// ========== RANDOMIZED SEARCH ==========
// var SVM_KERNEL = 'RBF';
// var SVM_C = 65.040;           // 65.04020246676566
// var SVM_GAMMA = 0.0002359;    // 0.0002359137306347712
// var SVM_EPSILON = 0.3055;     // 0.3054603435341466

var featureNames = [
  'lulc', 'Density_River', 'Density_Road', 'Distan2river', 'Distan2road_met',
  'aspect', 'curvature', 'dem', 'flowDir', 'slope', 'twi', 'NDVI', 'rainfall'
];

// FUNCTION: STANDARDIZATION (Z-SCORE)
var standardizeFeatures = function(data, featureNames) {
  var stats = featureNames.map(function(feature) {
    var values = data.aggregate_array(feature);
    var mean = values.reduce(ee.Reducer.mean());
    var stdDev = values.reduce(ee.Reducer.stdDev());
    return ee.Dictionary({
      'feature': feature,
      'mean': mean,
      'std': stdDev
    });
  });
  
  var statsDict = ee.Dictionary.fromLists(
    ee.List(stats).map(function(d) { return ee.Dictionary(d).get('feature'); }),
    stats
  );
  
  var standardized = data.map(function(f) {
    var props = {};
    featureNames.forEach(function(feature) {
      var stat = ee.Dictionary(statsDict.get(feature));
      var mean = ee.Number(stat.get('mean'));
      var std = ee.Number(stat.get('std'));
      var value = ee.Number(f.get(feature));
      var standardizedValue = ee.Algorithms.If(
        std.eq(0),
        0,
        value.subtract(mean).divide(std)
      );
      props[feature + '_std'] = standardizedValue;
    });
    return f.set(props);
  });
  
  return ee.Dictionary({
    'data': standardized,
    'stats': statsDict
  });
};

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

// STANDARDIZE DATA
var stdResult = standardizeFeatures(trainingData, featureNames);
var trainingDataStd = ee.FeatureCollection(ee.Dictionary(stdResult).get('data'));
var stdStats = ee.Dictionary(ee.Dictionary(stdResult).get('stats'));

var standardizedFeatureNames = featureNames.map(function(name) {
  return name + '_std';
});

var withRandom = trainingDataStd.randomColumn('random', 42);
var training = withRandom.filter(ee.Filter.lt('random', TRAIN_SPLIT));
var validation = withRandom.filter(ee.Filter.gte('random', TRAIN_SPLIT));

var svrModel = ee.Classifier.libsvm({
  svmType: 'EPSILON_SVR',
  kernelType: SVM_KERNEL,
  gamma: SVM_GAMMA,
  cost: SVM_C,
  lossEpsilon: SVM_EPSILON
}).setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'flood',
    inputProperties: standardizedFeatureNames
  });

// STANDARDIZE FEATURES FOR PREDICTION
var featuresStd = features;
featureNames.forEach(function(feature) {
  var stat = ee.Dictionary(stdStats.get(feature));
  var mean = ee.Number(stat.get('mean'));
  var std = ee.Number(stat.get('std'));
  var band = features.select([feature]);
  var standardizedBand = band.subtract(mean).divide(std.max(1e-10));
  featuresStd = featuresStd.addBands(
    standardizedBand.rename(feature + '_std'),
    null,
    true
  );
});

var floodsusceptibility = featuresStd.select(standardizedFeatureNames).classify(svrModel)
  .clamp(0, 1)
  .rename('flood_susceptibility')
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

// Classify toàn bộ ảnh features đã standardized
var predictions = featuresStd.select(standardizedFeatureNames).classify(svrModel).rename('prediction');

// Sample predictions tại các điểm validation
var validationPredictions = predictions.sampleRegions({
  collection: validationWithGeom,
  properties: ['flood', 'lat', 'lon'],
  scale: RESOLUTION,
  tileScale: 1,
  geometries: false
});

print('SVM Validation Count:', validationPredictions.size());
print('SVM Validation Sample:', validationPredictions.limit(10));

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
print('SVM R² Score:', r2Score);

// Export validation data để vẽ 2D histogram
Export.table.toDrive({
  collection: validationPredictions.select(['flood', 'prediction', 'lat', 'lon']),
  description: 'validation_SVM',
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
  description: 'flood_susceptibility_SVR',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  crs: 'EPSG:4326'
});
