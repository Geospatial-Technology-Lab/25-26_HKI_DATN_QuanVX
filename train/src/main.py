import os
import pandas as pd
import joblib
import numpy as np
from preprocess import preprocess_data
from train.src.predict_class import predict_flood_probabilities
from utils import create_all_tiffs, classify_flood_risk

def main():
    """
    Chương trình chính để dự báo lũ lụt và xuất kết quả
    """
    print("🌊 BẮT ĐẦU DỰ BÁO LŨ LỤT")
    print("=" * 50)
    
    # 1. Đọc và xử lý dữ liệu
    print("📊 Đang đọc dữ liệu...")
    try:
        X, coordinates = preprocess_data('data/flood_points.csv')
        print(f"✅ Đã tải {len(X):,} điểm dữ liệu")
        print(f"📐 Số features: {X.shape[1]}")
        
        # Tạo DataFrame cho coordinates
        coord_df = pd.DataFrame(coordinates, columns=['x', 'y'])
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc dữ liệu: {e}")
        return
    
    # 2. Tải và chạy các mô hình
    print("\n🤖 Đang chạy dự báo với các mô hình...")
    try:
        predictions_df = predict_flood_probabilities(
            X=X, 
            position_data=coord_df, 
            models_dir=r'D:\25-26_HKI_DATN_QuanVX\train\model'
        )
        print(f"✅ Hoàn thành dự báo cho {len(predictions_df):,} điểm")
    except Exception as e:
        print(f"❌ Lỗi khi dự báo: {e}")
        return
    
    # 3. Xuất kết quả thành raster
    print("\n🗺️ Đang xuất ảnh raster...")
    try:
        # Tìm các cột prediction
        prediction_cols = [col for col in predictions_df.columns if col.startswith('prob_')]
        
        if not prediction_cols:
            print("❌ Không tìm thấy cột dự báo nào!")
            return
        
        # Sử dụng hàm create_all_tiffs để tạo cả probability và risk maps
        success_count = create_all_tiffs(predictions_df, output_dir='output')
        
        if success_count == 0:
            print("❌ Không tạo được file nào!")
            return
        
    except Exception as e:
        print(f"❌ Lỗi khi xuất raster: {e}")
        return
    
    # 4. Báo cáo hoàn thành
    print("\n🎉 HOÀN THÀNH!")
    print(f"📁 Kết quả đã được lưu trong thư mục 'output/'")
    print("\nCác file đã tạo cho mỗi model:")
    print("  - flood_prob_[model_name].tif (xác suất lũ lụt 0.0-1.0)")
    print("  - flood_risk_[model_name].tif (phân loại rủi ro 1-5)")
    print(f"\n💡 Kết quả:")
    print(f"   - Xác suất: 0.0 = không lũ → 1.0 = chắc chắn có lũ")
    print(f"   - Rủi ro: 1=rất thấp, 2=thấp, 3=trung bình, 4=cao, 5=rất cao")
    print(f"   - Giá trị > 0.5 thường được coi là có khả năng xảy ra lũ")

if __name__ == "__main__":
    main()