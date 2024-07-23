import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs
import json
from time import sleep
from pprint import pformat
import threading
import os
import pandas as pd
from datetime import datetime

# Khởi tạo và cấu hình WebDriver
options = uc.ChromeOptions()
options.user_data_dir = "./ChromeProfile/02"
driver = uc.Chrome(version_main=124, options=options, enable_cdp_events=True)

# Các URL API cho việc truy xuất dữ liệu từ Shopee
shop_base_api = "https://shopee.vn/api/v4/shop/get_shop_base"
shop_item_rcmd_api = "https://shopee.vn/api/v4/shop/rcmd_items"
shop_sold_out_api = "https://shopee.vn/api/v4/shop/search_items?filter_sold_out"

# Số lượng sản phẩm trên một trang
items_per_page = 30

# Biến toàn cục để lưu trữ offset hiện tại
global offset
offset = 0

# Biến toàn cục để chờ dữ liệu hoàn tất
global wait_flag
wait_flag = 0 

# Hàm xử lý dữ liệu XHR được nhận
def handle_xhr_data(event_data):
    response_url = event_data["params"]["response"]["url"]
    if not response_url.startswith("https://shopee.vn/api/v4/shop/"):
        return
    
    request_id = event_data["params"]["requestId"]
    data = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
    if response_url.startswith(shop_base_api):
        convert_base_shop_info(data['body'])

    if response_url.startswith(shop_item_rcmd_api):
        data = json.loads(data['body'])['data']
        items = data['items']

        global offset
        offset += 1
        shop_data_dict['recommended'][offset] = items

        if collect_info['total_recommended_items'] == 0:
            print("Tổng số sản phẩm đề xuất:", data['total'])

        collect_info['total_recommended_items'] = data['total']

# Tính số trang dựa trên số lượng sản phẩm
def calculate_page_count(item_count):
    page_count = item_count // items_per_page 
    if item_count % items_per_page == 0:
        page_count -= 1
    return page_count 

# Hàm gọi khi trang web được điều hướng
def on_page_navigated(event_data):
    response_url = event_data["params"]["frame"]["url"]
    if response_url == 'about:blank':
        return

    print("Đã điều hướng đến trang:", response_url)

# Chuyển đổi thông tin cơ bản của shop từ dữ liệu thô
def convert_base_shop_info(raw_data):
    data = json.loads(raw_data)   
    name = data['data']['name']
    description = data['data']['description']
    item_count = data['data']['item_count']
    print(pformat({
        'Tên cửa hàng': name,
        'Mô tả': description,
        'Số lượng sản phẩm': item_count
    }))

    collect_info['total_items'] = item_count

    # Bắt đầu thu thập thông tin sản phẩm của shop
    collector_thread = threading.Thread(target=start_collecting_shop_items, args=(item_count,))
    collector_thread.start()

# Bắt đầu thu thập thông tin sản phẩm của shop
def start_collecting_shop_items(item_count):
    page_count = calculate_page_count(item_count)

    print("Tính toán số trang để thu thập dữ liệu:", page_count)
    
    for i in range(page_count):
        if collect_info['total_recommended_items'] != 0 and (i * items_per_page) > collect_info['total_recommended_items']:
            break

        collect_in_page(i)
    
    print("Hoàn tất quá trình thu thập !!!")

    save_collected_data()

# Thu thập thông tin sản phẩm trong một trang cụ thể
def collect_in_page(page_index):
    if page_index == 0:
        sleep(5)

    class_name = "shopee-button-no-outline"
    content = page_index + 1
    xpath_expression = f"//button[contains(@class, '{class_name}') and text()='{content}']"
    
    try:
        button = driver.find_element('xpath', xpath_expression)
        if button:
            print("Gửi yêu cầu click cho nút:", page_index)
            button.click()

    except:
        return

# Xử lý và lưu dữ liệu đã thu thập
def save_collected_data():
    global wait_flag
    folder_path = './data/raw/daily_crawl' + '/' + shop_id
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    all_items = []
    [all_items.extend(sub_items) for sub_items in shop_data_dict['recommended'].values()]

    keys_to_extract = ['itemid', 'shopid', 'liked_count', 'cmt_count', 'discount', 'shop_location', 'shop_rating',
                       'name', 'historical_sold', 'price']
    
    rating_columns = ['rating_star', 'rating_count', 'rcount_with_context']

    all_columns = keys_to_extract + rating_columns 

    df = pd.DataFrame(columns=all_columns)

    for item in all_items:
        row_data = {key: item.get(key, None) for key in keys_to_extract}
        row_data['rating_star'] = item.get('item_rating', {}).get('rating_star', None)
        row_data['rating_count'] = item.get('item_rating', {}).get('rating_count', None)
        row_data['rcount_with_context'] = item.get('item_rating', {}).get('rcount_with_context', None)

        row_df = pd.DataFrame([row_data])
        df = pd.concat([df, row_df], ignore_index=True)

    file_path = f'./data/raw/daily_crawl/{shop_id}/{datetime.now().strftime("%Y-%m-%d")}.csv'
    if os.path.exists(file_path):
        temp_df = pd.read_csv(file_path)        
        df = pd.concat([df, temp_df], ignore_index=True)

    df = df.drop_duplicates(subset=['itemid', 'shopid'], keep='first')
    df.to_csv(file_path, index=False)
    wait_flag = 1
    print("Lưu dữ liệu cho cửa hàng:", shop_id)
    print("Tổng số sản phẩm đề xuất:", collect_info['total_recommended_items'])
    print("Độ dài dữ liệu:", len(df))

# Bắt đầu quá trình thu thập dữ liệu cho cửa hàng
def crawl_shop(shop_id):
    global shop_data_dict, collect_info
    shop_data_dict = {"recommended": {}, "sold_out": {}}

    collect_info = {'total_items': 0, 'total_recommended_items': 0, 'total_sold_out_items': 0}
    shop_url = f'https://shopee.vn/{shop_id}#product_list'
    driver.add_cdp_listener("Network.responseReceived", handle_xhr_data)
    driver.add_cdp_listener("Page.frameNavigated", on_page_navigated)
    driver.get(shop_url)

# Danh sách ID cửa hàng để thu thập dữ liệu
shop_ids = ['bioderma_officialstore', 'cocoonvietnamofficial', 'drthem', 'garnier_officialstore', 'hadalabo.officialstore', 'hasaki.vn', 'lameila.official', 'larocheposay_officialstore', 'lorealparis_officialstore', 'nuty.vn', 'seovnpro', 'the_gioi_skin_food', 'unilevervn_beauty']

# Duyệt qua từng ID cửa hàng và bắt đầu quá trình thu thập
for shop_id in shop_ids:
    wait_flag = 0
    sleep(5)

    crawl_shop(shop_id)    

    while wait_flag == 0:
        sleep(3)

# Đóng WebDriver sau khi đã thu thập dữ liệu từ tất cả các cửa hàng
driver.quit()
