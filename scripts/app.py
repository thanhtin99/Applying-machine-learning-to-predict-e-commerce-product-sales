import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TABLEAU_COLORS
from tensorflow.keras.models import load_model

# Define functions to validate and check dates
def validate_date_format(date_string):
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def check_consecutive_dates(dates):
    dates.sort()
    for i in range(len(dates) - 1):
        if (dates[i + 1] - dates[i]).days != 1:
            return False
    return True

def check_same_dates_across_shops(shop_dates):
    first_shop_dates = sorted(shop_dates[0])
    for dates in shop_dates:
        if sorted(dates) != first_shop_dates:
            return False
    return True

# Initialize session state
if 'uploaded_files_per_shop' not in st.session_state:
    st.session_state.uploaded_files_per_shop = {}
if 'completed' not in st.session_state:
    st.session_state.completed = False
if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'ok_clicked' not in st.session_state:
    st.session_state.ok_clicked = False
if 'report_page' not in st.session_state:
    st.session_state.report_page = False
if 'num_days' not in st.session_state:
    st.session_state.num_days = 14
if 'num_shops' not in st.session_state:
    st.session_state.num_shops = 2
if 'selected_shop' not in st.session_state:
    st.session_state.selected_shop = None

st.title("Shop Sales Data Analysis")

if not st.session_state.report_page:
    st.sidebar.header("Settings")
    st.session_state.num_days = st.sidebar.number_input("Select number of days", min_value=1, value=14)
    st.session_state.num_shops = st.sidebar.number_input("Select number of shops", min_value=1, value=2)

    if st.sidebar.button("Initialize"):
        st.session_state.uploaded_files_per_shop = {shop: [] for shop in range(st.session_state.num_shops)}
        st.session_state.completed = False
        st.session_state.ok_clicked = True

    if st.session_state.ok_clicked:
        shop_options = [f"Shop {shop + 1}" for shop in range(st.session_state.num_shops)]
        selected_shop = st.selectbox("Select a shop", shop_options)
        st.session_state.selected_shop = shop_options.index(selected_shop)

        if st.session_state.selected_shop is not None:
            st.subheader(f"Upload files for {selected_shop}")
            uploaded_files = st.file_uploader(f"Choose {st.session_state.num_days} files for {selected_shop}", type=['csv'], accept_multiple_files=True, key=f"shop_{st.session_state.selected_shop}")
            if uploaded_files:
                st.session_state.uploaded_files_per_shop[st.session_state.selected_shop] = uploaded_files

    if st.session_state.ok_clicked and st.sidebar.button("Compute"):
        format_errors = []
        shop_dates = [[] for _ in range(st.session_state.num_shops)]
        all_dfs = []

        for shop in range(st.session_state.num_shops):
            uploaded_files = st.session_state.uploaded_files_per_shop.get(shop, [])
            
            if len(uploaded_files) != st.session_state.num_days:
                st.error(f"Shop {shop + 1}: Please upload exactly {st.session_state.num_days} files.")
            else:
                for file in uploaded_files:
                    file_name = file.name
                    date_part = file_name.split('.')[0]
                    
                    if validate_date_format(date_part):
                        date = datetime.strptime(date_part, '%Y-%m-%d')
                        shop_dates[shop].append(date)
                        
                        df = pd.read_csv(file)
                        df['date'] = date_part
                        all_dfs.append(df)
                    else:
                        format_errors.append(file_name)
        
        if format_errors:
            st.error(f"Files with incorrect date format: {', '.join(format_errors)}")
        else:
            all_dates_consecutive = all(check_consecutive_dates(dates) for dates in shop_dates)
            dates_same_across_shops = check_same_dates_across_shops(shop_dates)
            
            if all_dates_consecutive and dates_same_across_shops:
                st.session_state.completed = True
                st.success("All files have the correct format and the dates are consecutive and consistent across shops.")
                
                # Concatenate all DataFrames and store in session state
                st.session_state.final_df = pd.concat(all_dfs, ignore_index=True)
                st.session_state.report_page = True
                st.experimental_rerun()
            else:
                if not all_dates_consecutive:
                    st.error("The dates within a shop are not consecutive.")
                if not dates_same_across_shops:
                    st.error("The dates are not the same across all shops.")

if st.session_state.report_page:
    st.sidebar.header("Filter Data")
    df = st.session_state.final_df.copy()
    num_days = st.session_state.num_days
    
    # Add a text input for keyword
    keyword = st.sidebar.text_input("Enter a keyword to filter products by name", "").strip()
    
    # Filter out rows based on keywords in 'name' column if keyword is provided
    if keyword:
        df = df[df['name'].str.contains(keyword, case=False, regex=False)]
    
    # Continue with existing filtering and processing
    keywords_pattern = r'\[HB GIFT\]|\[Quà tặng\]|\[Gift\]|\[HC Gift\]|không bán|\[Tặng kèm\]'
    df = df[~df['name'].str.contains(keywords_pattern, case=False, regex=True)]
    
    # Fill missing values in 'discount' column with '0%'
    df['discount'].fillna('0%', inplace=True)
    
    # Calculate 'count_date' column
    df['count_date'] = df.groupby('itemid')['date'].transform('count')
    
    # Filter rows where 'count_date' equals num_days
    df = df[df['count_date'] == num_days]
    
    # Convert 'discount' to integer
    df['discount'] = df['discount'].str.replace('%', '').astype(int)
    
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by 'itemid' and 'date'
    df.sort_values(by=['itemid', 'date'], inplace=True)
    
    # Calculate daily sales
    df['daily_sales'] = df.groupby('itemid')['historical_sold'].diff()

    # Tính median daily_sales cho mỗi itemid
    median_sales = df[df['daily_sales'] >= 0].groupby('itemid')['daily_sales'].median()

    # Tạo một hàm để áp dụng median
    def replace_with_median(row):
        if row['daily_sales'] < 0:
            return median_sales[row['itemid']]
        return row['daily_sales']

    # Áp dụng hàm để thay thế giá trị daily_sales
    df['daily_sales'] = df.apply(replace_with_median, axis=1)

    # Filter by price
    min_price, max_price = st.sidebar.slider(
        "Select price range",
        float(df['price'].min()/100000), 
        float(df['price'].max()/100000), 
        (float(df['price'].min()/100000), float(df['price'].max()/100000))
    )
    df = df[(df['price'] >= min_price*100000) & (df['price'] <= max_price*100000)]



    # Create lagged sales and price columns
    df['price'] = df['price']/100000
    
    WINDOW_SIZE = 7
    for i in range(1, WINDOW_SIZE + 1):
        df[f'daily_sales_lag_{i}d'] = df.groupby('itemid')['daily_sales'].shift(i)
        df[f'price_lag_{i}d'] = df.groupby('itemid')['price'].shift(i)

    # Fill missing values in new columns with median values
    new_columns = [f'daily_sales_lag_{i}d' for i in range(1, WINDOW_SIZE + 1)] + [f'price_lag_{i}d' for i in range(1, WINDOW_SIZE + 1)]
    df[new_columns] = df.groupby('itemid')[new_columns].transform(lambda x: x.fillna(x.median()))
        
    # Calculate rolling mean of daily sales
    df[f'rolling_mean_{WINDOW_SIZE}d'] = df.groupby('itemid')['daily_sales'].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean())
    
    # Combine with sentiment analysis data
    agg_table = pd.read_csv('../data/processed/agg_table.csv')
    agg_table['date'] = pd.to_datetime(agg_table['date'])
    df = pd.merge(df, agg_table, on=['itemid', 'date'], how='left')
    df.sort_values(by=['itemid', 'date'], inplace=True)
    
    # Forward fill sentiment analysis columns
    df['label_1_count'] = df.groupby('itemid')['label_1_count'].bfill()
    df['label_1_count'] = df.groupby('itemid')['label_1_count'].ffill()
    df.dropna(subset=['label_1_count'], inplace=True)
    
    # Select final columns
    df['itemid'] = df['itemid'].astype(str)
    df['shopid'] = df['shopid'].astype(str)
    st.write(df)

    # Define a color palette
    colors = list(TABLEAU_COLORS.values())

    shop_name_df = pd.read_csv("../data/raw/shop_name.csv")
    shop_name_df['shopid'] = shop_name_df['shopid'].astype(str) 
    merged_df = pd.merge(df, shop_name_df, on='shopid')

    st.subheader("Exploratory Data Analysis (EDA)")

    # EDA: Distribution of Products With and Without Discount Across Shops
    st.write("### Distribution of Products With and Without Discount Across Shops")
    discount_distribution = merged_df.groupby(['shop_name', merged_df['discount'] > 0])['discount'].count().unstack().fillna(0)

    # Check if the columns are as expected and rename accordingly
    if discount_distribution.shape[1] == 2:
        discount_distribution.columns = ['No Discount', 'With Discount']
    else:
        discount_distribution.columns = [f'Column {i}' for i in range(discount_distribution.shape[1])]

    # Plotting
    discount_distribution.plot(kind='bar', stacked=True, figsize=(14, 10), color=sns.color_palette('Set2', discount_distribution.shape[1]))
    plt.title('Distribution of Products With and Without Discount Across Shops')
    plt.xlabel('Shop Name')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45)
    plt.legend(title='Discount Status')
    plt.tight_layout()

    st.pyplot(plt)

    # Metrics to compare
    st.write("### Average Discount and Price per Shop")
    metrics_avg = ['discount', 'price']

    # Bar charts for average metrics
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics_avg, 1):
        avg_metric = merged_df.groupby('shop_name')[metric].mean().reset_index()
        plt.subplot(1, 2, i)
        sns.barplot(data=avg_metric, x='shop_name', y=metric, palette='Set2', ci=None)
        plt.title(f'Average {metric.replace("_", " ").title()} per Shop')
        plt.xlabel('Shop Name')
        plt.ylabel(f'Average {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Time Series Charts
    st.subheader("Time Series Charts")

    st.write("### Change in Discount and Daily Sales over Time for Each Shop")
    # Line chart: Change in discount over time for each shop
    metrics = ['discount', 'daily_sales']
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 1, i)
        sns.lineplot(data=merged_df, x='date', y=metric, hue='shop_name', palette='Set2')
        plt.title(f'{metric.replace("_", " ").title()} over Time')
        plt.xlabel('Date')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Preparing data for the model
    df_for_model = df[[
        'itemid', 
        'date',
        'shopid',
        'historical_sold',
        'price',
        'rating_count',
        'rcount_with_context',
        'daily_sales',
        'daily_sales_lag_1d',
        'price_lag_1d',
        'daily_sales_lag_2d',
        'price_lag_2d',
        'daily_sales_lag_3d',
        'price_lag_3d',
        'daily_sales_lag_4d',
        'price_lag_4d',
        'daily_sales_lag_5d',
        'price_lag_5d',
        'daily_sales_lag_6d',
        'price_lag_6d',
        'daily_sales_lag_7d',
        'price_lag_7d',
        'rolling_mean_7d',
        'label_1_count'
    ]]

    SEQ_LEN = 7

    # Load models
    lstm_model = load_model('../data/models/lstm_dropout_model.keras')
    model_lr = joblib.load(r'../data/models/model_lr.pkl')
    model_rf = joblib.load(r'../data/models/model_rf.pkl')
    model_xgb = joblib.load(r'../data/models/model_xgb.pkl')
    model_knn = joblib.load(r'../data/models/model_knn.pkl')
    model_dt = joblib.load(r'../data/models/model_dt.pkl')
    scaler_lstm = joblib.load('../data/models/scaler_lstm.pkl')
    scaler_y_lstm = joblib.load('../data/models/scaler_y_lstm.pkl')
    scaler = joblib.load('../data/models/scaler.pkl')

    def has_sufficient_data(itemid):
        item_data = df_for_model[df_for_model['itemid'] == itemid]
        item_data = item_data.drop(columns=['itemid', 'shopid', 'date'])
        item_data = item_data.dropna()
        
        if len(item_data) < SEQ_LEN:
            return False

        return True

    itemid_options = df_for_model['itemid'].unique().tolist()
    sufficient_data_itemids = [itemid for itemid in itemid_options if has_sufficient_data(itemid)]

    # Default selection: itemid with max sum(daily_sales) per shopid
    default_selected_itemids = df_for_model[df_for_model['itemid'].isin(sufficient_data_itemids)].groupby('shopid')['daily_sales'].idxmax()

    # Ensure default_selected_itemids is a list and contains valid itemids
    if not isinstance(default_selected_itemids, list):
        default_selected_itemids = default_selected_itemids.tolist()
    default_selected_itemids = [itemid for itemid in default_selected_itemids if itemid in sufficient_data_itemids]

    selected_itemids = st.multiselect("Select itemid", sufficient_data_itemids, default_selected_itemids)

    # Filter DataFrame based on selected itemid
    df_for_model_filtered = df_for_model[df_for_model['itemid'].isin(selected_itemids)]

    # Select model for prediction
    model_options = ['LSTM', 'Linear Regression', 'Random Forest', 'XGBoost', 'KNN', 'Decision Tree']
    selected_model = st.selectbox("Select model for prediction", model_options)

    st.subheader("Revenue Comparison")

    def get_model(model_name):
        if model_name == 'LSTM':
            return lstm_model
        elif model_name == 'Linear Regression':
            return model_lr
        elif model_name == 'Random Forest':
            return model_rf
        elif model_name == 'XGBoost':
            return model_xgb
        elif model_name == 'KNN':
            return model_knn
        elif model_name == 'Decision Tree':
            return model_dt
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def predict_sales(model, X_input_scaled):
        if selected_model == 'LSTM':
            y_pred = model.predict(X_input_scaled)
            y_pred = scaler_y_lstm.inverse_transform(y_pred).flatten()
        else:
            X_input_flattened = X_input_scaled.reshape(X_input_scaled.shape[0], -1)
            y_pred = model.predict(X_input_flattened)
        return y_pred

    def plot_revenue_comparison():
        plt.figure(figsize=(10, 6))
        if selected_model == 'LSTM':
            for idx, itemid in enumerate(selected_itemids):
                item_data = df_for_model_filtered[df_for_model_filtered['itemid'] == itemid]
                color = colors[idx % len(colors)]
                plt.plot(item_data['date'], item_data['daily_sales'], label=f'Actual Sales for Item {itemid}', color=color)
                
                # Prepare data for prediction
                last_7_days = item_data.tail(SEQ_LEN).drop(columns=['itemid', 'shopid', 'date'])
                X_input = last_7_days.values.reshape(1, SEQ_LEN, -1)

                X_input_scaled = scaler_lstm.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)
                y_pred = predict_sales(get_model(selected_model), X_input_scaled)
                
                next_day = item_data['date'].max() + pd.Timedelta(days=1)
                
                plt.plot(next_day, y_pred, marker='o', markersize=10, linestyle='', color=color, label=f'Predicted Sales for Item {itemid}')
            
        else:
            for idx, itemid in enumerate(selected_itemids):
                item_data = df_for_model_filtered[df_for_model_filtered['itemid'] == itemid]
                color = colors[idx % len(colors)]
                plt.plot(item_data['date'], item_data['daily_sales'], label=f'Actual Sales for Item {itemid}', color=color)
                
                # Prepare data for prediction
                last_7_days = item_data.tail(SEQ_LEN).drop(columns=['itemid', 'shopid', 'date'])
                X_input = item_data.tail(1).drop(columns=['itemid', 'shopid', 'date']).values
                X_input_scaled = scaler.transform(X_input)
                y_pred = predict_sales(get_model(selected_model), X_input_scaled)
                
                next_day = item_data['date'].max() + pd.Timedelta(days=1)
                
                plt.plot(next_day, y_pred, marker='o', markersize=10, linestyle='', color=color, label=f'Predicted Sales for Item {itemid}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Daily Sales and Predicted Sales Comparison')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        st.pyplot(plt)

    # Plot comparison
    if st.button("Plot Comparison"):
        plot_revenue_comparison()
