import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the features (X)
df = pd.read_csv('https://raw.githubusercontent.com/chandrugtx8/housingsales/main/Flat%20prices.csv')
df['month'] = pd.to_datetime(df['month'])
df['year'] = df['month'].dt.year
df['month_of_year'] = df['month'].dt.month
df['remaining_lease_years'] = df['remaining_lease'].apply(lambda x: int(x.split()[0]))
df.drop(columns=['month', 'remaining_lease', 'block', 'street_name'], inplace=True)
X = pd.get_dummies(df, columns=['town', 'flat_type', 'storey_range', 'flat_model']).drop(columns=['resale_price'])

def train_model(X):
    # Train a RandomForestRegressor model
    model = RandomForestRegressor()
    y = df['resale_price']
    model.fit(X, y)
    return model

def main(X):
    st.title('Housing Sales Prediction')

    town_options = df['town'].unique()
    town = st.selectbox("Select the town:", town_options)

    flat_type_options = df['flat_type'].unique()
    flat_type = st.selectbox("Select the flat type:", flat_type_options)

    storey_range_options = df['storey_range'].unique()
    storey_range = st.selectbox("Select the storey range:", storey_range_options)

    floor_area_sqm_str = st.text_input("Enter the floor area (in sqm): ")
    st.write("Floor area input:", floor_area_sqm_str)  # Debug statement

    flat_model_options = df['flat_model'].unique()
    flat_model = st.selectbox("Select the flat model:", flat_model_options)

    lease_commence_date_str = st.text_input("Enter the lease commence date: ")
    st.write("Lease commence date input:", lease_commence_date_str)  # Debug statement

    if st.button('Predict'):
        try:
            floor_area_sqm = float(floor_area_sqm_str)
            lease_commence_date = int(lease_commence_date_str)
            input_data = pd.DataFrame({
                'town': [town],
                'flat_type': [flat_type],
                'storey_range': [storey_range],
                'floor_area_sqm': [floor_area_sqm],
                'flat_model': [flat_model],
                'lease_commence_date': [lease_commence_date]
            })

            # Ensure input data is encoded in the same way as training data
            input_data_encoded = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)

            # Train the model
            model = train_model(X)

            # Make prediction
            predicted_price = model.predict(input_data_encoded)[0]
            st.success('Predicted Resale Price: {:.2f}'.format(predicted_price))
        except ValueError:
            st.error("Please enter valid numeric values for floor area and lease commence date.")

if __name__ == '__main__':
    main(X)
