import pandas as pd
import numpy as np

def chuan_hoa_du_lieu(file_path):
    """
    Chuẩn hóa dữ liệu CSV theo công thức Min-Max: (x - xmin) / (xmax - xmin)
    
    Args:
        file_path: Đường dẫn đến file CSV cần chuẩn hóa
    """
    # Đọc file CSV
    print(f"Đang đọc file: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Số dòng: {len(df)}, Số cột: {len(df.columns)}")
    print(f"Tên các cột: {list(df.columns)}")
    
    # Lấy tên cột đầu tiên (không chuẩn hóa)
    first_column = df.columns[0]
    print(f"\nCột '{first_column}' sẽ không được chuẩn hóa")
    
    # Chuẩn hóa các cột còn lại
    columns_to_normalize = df.columns[1:]
    print(f"Các cột cần chuẩn hóa: {list(columns_to_normalize)}")
    
    for col in columns_to_normalize:
        # Kiểm tra xem cột có phải là số không
        if pd.api.types.is_numeric_dtype(df[col]):
            x_min = df[col].min()
            x_max = df[col].max()
            
            # Tránh chia cho 0
            if x_max - x_min != 0:
                df[col] = (df[col] - x_min) / (x_max - x_min)
                print(f"  - Chuẩn hóa cột '{col}': min={x_min:.4f}, max={x_max:.4f}")
            else:
                print(f"  - Bỏ qua cột '{col}' (tất cả giá trị giống nhau: {x_min})")
        else:
            print(f"  - Bỏ qua cột '{col}' (không phải dữ liệu số)")
    
    # Ghi đè lên file gốc
    print(f"\nĐang ghi đè kết quả lên file: {file_path}")
    df.to_csv(file_path, index=False)
    print("Hoàn thành chuẩn hóa dữ liệu!")
    
    # Hiển thị vài dòng đầu tiên sau khi chuẩn hóa
    print("\nDữ liệu sau khi chuẩn hóa (5 dòng đầu):")
    print(df.head())

if __name__ == "__main__":
    # Đường dẫn file đầu vào
    input_file = r"D:\25-26_HKI_DATN_QuanVX\train\data\training_data.csv"
    
    # Thực hiện chuẩn hóa
    chuan_hoa_du_lieu(input_file)
