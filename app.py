import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from datetime import datetime, timedelta
import networkx as nx
from scipy.stats import pearsonr
from scipy.signal import detrend

# Google Gemini API integration
GEMINI_API_KEY = "enter your gemini api here"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

def query_gemini(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    params = {'key': GEMINI_API_KEY}
    
    response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'date' not in df.columns:
        df['date'] = pd.date_range(start='2023-01-01', periods=len(df))
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    # Remove rows with NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

def sales_forecast(data, target_column, forecast_period=30):
    X = pd.DataFrame({'date_num': range(len(data))})
    y = data[target_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    future_dates = pd.DataFrame({'date_num': range(len(data), len(data) + forecast_period)})
    forecast = model.predict(future_dates)
    
    return forecast

def simple_moving_average(data, window=7):
    return data.rolling(window=window).mean()

def churn_prediction(data, features, target):
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, accuracy, model.feature_importances_

def customer_segmentation(data, features, n_clusters=3):
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    silhouette_avg = silhouette_score(X_scaled, clusters)
    
    return clusters, silhouette_avg

def generate_insights(data):
    insights = []
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        for col in numeric_columns:
            trend = data[col].pct_change().mean()
            insights.append(f"{col} trend: {'Increasing' if trend > 0 else 'Decreasing'} by {abs(trend):.2%} on average")
    
    for col in data.columns:
        if data[col].dtype == 'object':
            top_values = data[col].value_counts().head(5)
            insights.append(f"Top 5 {col}: {', '.join([f'{v} ({c})' for v, c in top_values.items()])}")
    
    return insights

def create_network_graph(data, source_col, target_col, value_col):
    G = nx.from_pandas_edgelist(data, source=source_col, target=target_col, edge_attr=value_col)
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]} - # of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def create_dashboard(data):
    st.title("Advanced AI-Driven Business Analytics Platform")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    target_column = st.selectbox("Select target column for analysis", numeric_columns)
    
    st.subheader(f"{target_column} Overview")
    fig = px.line(data, x='date', y=target_column, title=f"{target_column} Over Time")
    st.plotly_chart(fig)
    
    st.subheader(f"{target_column} Forecast")
    forecast_method = st.radio("Select forecasting method", ["Random Forest", "Simple Moving Average"])
    forecast_period = st.slider("Forecast period (days)", 7, 90, 30)
    
    if forecast_method == "Random Forest":
        forecast = sales_forecast(data, target_column, forecast_period)
    else:
        forecast = simple_moving_average(data[target_column], window=7).iloc[-forecast_period:]
    
    future_dates = pd.date_range(start=data['date'].iloc[-1] + pd.Timedelta(days=1), periods=len(forecast))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data[target_column], name='Historical'))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, name='Forecast'))
    fig.update_layout(title=f"{target_column} Forecast", xaxis_title="Date", yaxis_title=target_column)
    st.plotly_chart(fig)
    
    st.subheader("Time Series Decomposition")
    try:
        trend = simple_moving_average(data[target_column], window=30)
        seasonal = data[target_column] - trend
        residual = detrend(seasonal.dropna())
        
        fig = make_subplots(rows=4, cols=1, subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'))
        fig.add_trace(go.Scatter(x=data['date'], y=data[target_column], name='Observed'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=trend, name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['date'], y=seasonal, name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['date'][:len(residual)], y=residual, name='Residual'), row=4, col=1)
        fig.update_layout(height=900, title_text="Time Series Decomposition")
        st.plotly_chart(fig)
    except Exception as e:
        st.write(f"Error in time series decomposition: {str(e)}")
    
    if 'churned' in data.columns:
        st.subheader("Customer Churn Prediction")
        churn_features = st.multiselect("Select features for churn prediction", numeric_columns)
        if churn_features:
            model, accuracy, feature_importance = churn_prediction(data, churn_features, 'churned')
            st.write(f"Churn prediction model accuracy: {accuracy:.2%}")
            
            fig = px.bar(x=churn_features, y=feature_importance, 
                         title="Feature Importance for Churn Prediction")
            st.plotly_chart(fig)
    
    st.subheader("Customer Segmentation")
    segmentation_features = st.multiselect("Select features for customer segmentation", numeric_columns)
    if segmentation_features:
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        clusters, silhouette = customer_segmentation(data, segmentation_features, n_clusters)
        st.write(f"Silhouette Score: {silhouette:.2f}")
        
        data['Cluster'] = clusters
        fig = px.scatter_3d(data, x=segmentation_features[0], y=segmentation_features[1], z=segmentation_features[2],
                            color='Cluster', title="3D Customer Segmentation")
        st.plotly_chart(fig)
    
    st.subheader("Network Analysis")
    source_col = st.selectbox("Select source column", data.columns)
    target_col = st.selectbox("Select target column", data.columns)
    value_col = st.selectbox("Select value column", numeric_columns)
    
    if source_col and target_col and value_col:
        network_fig = create_network_graph(data, source_col, target_col, value_col)
        st.plotly_chart(network_fig)
    
    st.subheader("Automated Insights")
    insights = generate_insights(data)
    for insight in insights:
        st.write(f"• {insight}")
    
    st.subheader("Ask AI Assistant")
    user_query = st.text_input("Ask a question about your business data:")
    if user_query:
        answer = query_gemini(f"Based on the business data and insights provided, {user_query}")
        st.write(answer)

    st.subheader("Interactive Data Explorer")
    selected_columns = st.multiselect("Select columns to display", data.columns)
    if selected_columns:
        st.dataframe(data[selected_columns].style.highlight_max(axis=0))

    st.subheader("Correlation Heatmap")
    if len(numeric_columns) > 1:
        corr_matrix = data[numeric_columns].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig)
    else:
        st.write("Not enough numeric columns for correlation analysis.")
    
    st.subheader("Custom Visualization")
    x_axis = st.selectbox("Select X-axis", data.columns)
    y_axis = st.selectbox("Select Y-axis", numeric_columns)
    chart_type = st.selectbox("Select chart type", ["Scatter", "Line", "Bar", "Box", "Violin", "Area", "Histogram"])
    color_by = st.selectbox("Color by (optional)", ["None"] + list(data.columns))
    
    if chart_type == "Scatter":
        fig = px.scatter(data, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by)
    elif chart_type == "Line":
        fig = px.line(data, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by)
    elif chart_type == "Bar":
        fig = px.bar(data, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by)
    elif chart_type == "Box":
        fig = px.box(data, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by)
    elif chart_type == "Violin":
        fig = px.violin(data, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by)
    elif chart_type == "Area":
        fig = px.area(data, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by)
    else:  # Histogram
        fig = px.histogram(data, x=x_axis, y=y_axis, color=None if color_by == "None" else color_by)
    
    fig.update_layout(title=f"Custom {chart_type} Chart: {y_axis} vs {x_axis}")
    st.plotly_chart(fig)

    st.subheader("Correlation Analysis")
    col1, col2 = st.columns(2)
    var1 = col1.selectbox("Select first variable", numeric_columns)
    var2 = col2.selectbox("Select second variable", numeric_columns)
    
    if var1 != var2:
        corr, _ = pearsonr(data[var1], data[var2])
        st.write(f"Pearson correlation between {var1} and {var2}: {corr:.2f}")
        
        fig = px.scatter(data, x=var1, y=var2, trendline="ols")
        fig.update_layout(title=f"Scatter plot: {var2} vs {var1}")
        st.plotly_chart(fig)
    else:
        st.write("Please select different variables for correlation analysis.")

    # Data Distribution
    st.subheader("Data Distribution")
    dist_var = st.selectbox("Select variable for distribution analysis", numeric_columns)
    
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Histogram(x=data[dist_var], name="Histogram"), row=1, col=1)
    fig.add_trace(go.Box(y=data[dist_var], name="Box Plot"), row=1, col=2)
    fig.update_layout(title=f"Distribution of {dist_var}")
    st.plotly_chart(fig)
    
    # Descriptive statistics
    st.write(data[dist_var].describe())

    # Anomaly Detection
    st.subheader("Anomaly Detection")
    anomaly_var = st.selectbox("Select variable for anomaly detection", numeric_columns)
    threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
    
    z_scores = (data[anomaly_var] - data[anomaly_var].mean()) / data[anomaly_var].std()
    anomalies = data[abs(z_scores) > threshold]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data[anomaly_var], mode='lines', name='Normal'))
    fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies[anomaly_var], mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
    fig.update_layout(title=f"Anomaly Detection for {anomaly_var}")
    st.plotly_chart(fig)
    
    st.write(f"Number of anomalies detected: {len(anomalies)}")
    if not anomalies.empty:
        st.write("Anomalies:")
        st.dataframe(anomalies)

    # Trend Analysis
    st.subheader("Trend Analysis")
    trend_var = st.selectbox("Select variable for trend analysis", numeric_columns)
    window = st.slider("Moving average window", 1, 30, 7)
    
    data['MA'] = data[trend_var].rolling(window=window).mean()
    data['Trend'] = np.where(data['MA'].diff() > 0, 'Increasing', 'Decreasing')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data[trend_var], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=data['date'], y=data['MA'], mode='lines', name=f'{window}-day MA'))
    fig.update_layout(title=f"Trend Analysis for {trend_var}")
    st.plotly_chart(fig)
    
    trend_counts = data['Trend'].value_counts()
    st.write("Trend Summary:")
    st.write(f"Increasing: {trend_counts.get('Increasing', 0)} days")
    st.write(f"Decreasing: {trend_counts.get('Decreasing', 0)} days")

    # Seasonality Detection
    st.subheader("Seasonality Detection")
    season_var = st.selectbox("Select variable for seasonality detection", numeric_columns)
    
    data['Day'] = data['date'].dt.day_name()
    data['Month'] = data['date'].dt.month_name()
    
    daily_avg = data.groupby('Day')[season_var].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    monthly_avg = data.groupby('Month')[season_var].mean().reindex(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Daily Seasonality", "Monthly Seasonality"))
    fig.add_trace(go.Bar(x=daily_avg.index, y=daily_avg.values, name="Daily Avg"), row=1, col=1)
    fig.add_trace(go.Bar(x=monthly_avg.index, y=monthly_avg.values, name="Monthly Avg"), row=1, col=2)
    fig.update_layout(title=f"Seasonality Analysis for {season_var}")
    st.plotly_chart(fig)

def main():
    st.sidebar.title("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            create_dashboard(data)
        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")
    else:
        st.write("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
