import pandas as pd
import os
import glob

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = './data/raw/daily_crawl'

# Biểu thức glob để lấy tất cả các tập tin .csv
file_pattern = os.path.join(data_dir, '*', '*.csv')

# Tìm tất cả các tập tin phù hợp
file_list = glob.glob(file_pattern)

# Khởi tạo một DataFrame rỗng để gộp dữ liệu
consolidated_data = pd.DataFrame()

# Xử lý từng tập tin
for file in file_list:
    # Đọc dữ liệu từ tập tin CSV
    df = pd.read_csv(file)
    
    # Trích xuất ngày tháng từ tên tập tin (ví dụ: yyyy-mm-dd.csv)
    date_part = os.path.basename(file).split('.')[0]
    
    # Thêm cột 'Date' với giá trị ngày tháng trích xuất
    df['date'] = pd.to_datetime(date_part)
    
    # Thêm dữ liệu vào DataFrame tổng hợp
    consolidated_data = pd.concat([consolidated_data, df], ignore_index=True)

# Lưu DataFrame tổng hợp vào tập tin CSV mới
output_file = './data/processed/merged_daily_data.csv'
consolidated_data.to_csv(output_file, index=False)

print('Dữ liệu đã được gộp và lưu thành công vào', output_file)
