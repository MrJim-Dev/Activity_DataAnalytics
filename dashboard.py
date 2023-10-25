import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
st.set_page_config(layout="wide")
# Load dataset
df = pd.read_csv("datasets/activities.csv")

# Check if 'selected_menu' is already in the session state
if 'selected_menu' not in st.session_state:
    selected_menu = "Overview"

# Style the buttons to make them full-width
st.markdown("""
    <style>
        .stButton > button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Style the buttons to make them full-width and add styling for custom metric container
st.markdown("""
    <style>
        .main {
            padding-left: 10%;
            padding-right: 10%;
        }
        .main > div {
            margin-top: -60px;
            text-align: center;
            display: flex;
            justify-content: center;
        }
        .stButton > button {
            width: 100%;
        }

        .customMetric {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 20px;
            margin: 1%;
            font-size: 1.2em;
            text-align: center;
            width: 23%;  /* Adjusted to 50% width */
            float: left;  /* Make them float left */
        }

        .customMetric h2 {
            color: black;
            font-weight: bold;
            margin: 0;
            padding: 0;
        }

        .customMetric h5 {
            color: black;
        }

        .customMetric p {
            color: black;
        }

        .customMetric label {
            color: #888;
            font-size: 0.9em;
            display: block;
            margin-top: 10px;
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar navigation using buttons
menu_options = ["Overview", "Personal", "Academics", "Work / Business", "Entertainment"]


# 2. horizontal menu
selected_menu = option_menu(None, ["Overview", "Personal", "Academics", "Work / Business", "Entertainment"], 
    icons=['house', 'person', "mortarboard", 'person-workspace',  'controller'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

st.write("\n")


# Display content based on selected menu
if selected_menu == "Overview":
    st.header("Activity Overview")
    st.write("###### by James Alein Ocampo")
    st.markdown("""
    *A comprehensive breakdown of my daily activities over the past two weeks.*

    This dashboard offers a glimpse into how I spent my time across different categories. It serves as a tool for reflection on my priorities and time management strategies.

    """)
    st.write("\n")

    st.markdown("<h2>Overview of the Categories</h2>", unsafe_allow_html=True)
    st.write("\n")
    

    # Compute the total hours spent for each category
    total_duration = df.groupby('Category')['Duration (hrs)'].sum()
    unique_days = df.groupby('Category')['Date'].nunique()
    avg_per_day = (total_duration / unique_days).round(2)
    # print(avg_per_day)

    metrics_html = ""  # Collecting metrics HTML in this string

    metrics_html += f"""
    <div class="customMetric">
        <h5>Personal</h5>
        <h2>{avg_per_day['Personal']}</h2>
        <label>Average Hours Spent</label>
    </div>
    """

    metrics_html += f"""
    <div class="customMetric">
        <h5>Academics</h5>
        <h2>{avg_per_day['Academics']}</h2>
        <label>Average Hours Spent</label>
    </div>
    """
    metrics_html += f"""
    <div class="customMetric">
        <h5>Work / Business</h5>
        <h2>{avg_per_day['Work / Business']}</h2>
        <label>Average Hours Spent</label>
    </div>
    """
    metrics_html += f"""
    <div class="customMetric">
        <h5>Entertainment</h5>
        <h2>{avg_per_day['Entertainment']}</h2>
        <label>Average Hours Spent</label>
    </div>
    """

    # Render the metrics
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.markdown('<hr/>', unsafe_allow_html=True);

    # Create a new 'Date_Day' column combining 'Date' and the first 3 letters of 'Day'
    df['Date & Day'] = df['Date'] + ' (' + df['Day'].str[:3] + ')'

    # Create two columns for the plots
    col1, middle_space, col2 = st.columns([1, 0.1, 1])  # Adjust the middle space as needed (0.1 for 10% width)

    # Plot the pie chart in the first column with some margin/padding
    with col1:
        category_duration = df.groupby('Category')['Duration (hrs)'].sum().reset_index()
        fig = px.pie(category_duration, values='Duration (hrs)', names='Category', title='Percentage Breakdown of Time Spent by Category')
        st.plotly_chart(fig, use_container_width=True)

    # Add a little space in the middle
    with middle_space:
        pass

    # Plot the bar chart in the second column with some margin/padding
    with col2:
        fig = px.bar(df, x='Date & Day', y='Duration (hrs)', color='Category', title='Duration of Activities by Category over Days')
        st.plotly_chart(fig, use_container_width=True)

elif selected_menu == "Personal":
    
    st.header("Personal Activities")
    st.write("Dive deeper into the analysis of my personal activities, revealing patterns in essential tasks and leisure moments. This section sheds light on my personal time is divided and prioritized over the course of two weeks.")
    st.write("\n")

    # Filter by category
    df_personal = df[df["Category"] == "Personal"]

    # Compute the total hours spent for each activity under the Personal category
    total_duration = df_personal.groupby('Activity')['Duration (hrs)'].sum()
    unique_days = df_personal.groupby('Activity')['Date'].nunique()
    avg_per_day = (total_duration / unique_days).round(2)

    # Collecting metrics HTML in this string
    metrics_html = ""

    for activity, avg_duration in avg_per_day.items():
        metrics_html += f"""
        <div class="customMetric">
            <h5>{activity}</h5>
            <h2>{avg_duration}</h2>
            <label>Average Hours Spent</label>
        </div>
        """

    # Render the metrics
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.markdown('<hr/>', unsafe_allow_html=True);

    df_personal['Date & Day'] = df_personal['Date'] + ' (' + df_personal['Day'].str[:3] + ')'

    # Create two columns for the plots
    col1, middle_space, col2 = st.columns([1, 0.1, 1])  # Adjust the middle space as needed (0.1 for 10% width)

    # Plot the pie chart in the first column with some margin/padding
    with col1:
        # Compute the total hours spent for each activity under the Personal category
        total_duration = df_personal.groupby('Activity')['Duration (hrs)'].sum().reset_index()

        # Create the pie chart
        pie_fig = px.pie(total_duration, names='Activity', values='Duration (hrs)', title='Total Hours for Each Personal Activity')
        st.plotly_chart(pie_fig)

    # Add a little space in the middle
    with middle_space:
        pass

    # Plot the bar chart in the second column with some margin/padding
    with col2:
        # Filter out the rows related to "Sleeping Time"
        sleep_df = df[df['Activity'] == 'Sleeping Time']

        # Getting the frequency of each activity within the 'Personal' category
        grouped = df_personal['Activity'].value_counts().reset_index()
        grouped.columns = ['Activity', 'Frequency']

        # Plotting the data using plotly
        fig = px.bar(grouped, x='Activity', y='Frequency', title='Frequency of Each Personal Activity')

        st.plotly_chart(fig)

    
    fig = px.bar(df_personal, x='Date & Day', y='Duration (hrs)', color='Activity', title='Duration of Personal Activities over Days')
    st.plotly_chart(fig)


elif selected_menu == "Academics":
    st.header("Academic Activities")
    st.write("Dive into the detailed analysis of how I spend an average day across various academic tasks, from attending classes to studying and managing tasks. Understand the distribution of my time and the frequency of each activity to gain insights into the intricacies of my academic life.")
    st.write("\n")
    # Filter by category
    df_academics = df[df["Category"] == "Academics"]
    
    # Compute the total hours spent for each activity under the Personal category
    total_duration = df_academics.groupby('Activity')['Duration (hrs)'].sum()
    unique_days = df_academics.groupby('Activity')['Date'].nunique()
    avg_per_day = (total_duration / unique_days).round(2)

    # Collecting metrics HTML in this string
    metrics_html = ""

    for activity, avg_duration in avg_per_day.items():
        metrics_html += f"""
        <div class="customMetric">
            <h5>{activity}</h5>
            <h2>{avg_duration}</h2>
            <label>Average Hours Spent</label>
        </div>
        """

    # Render the metrics
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.markdown('<hr/>', unsafe_allow_html=True);

    df_academics['Date & Day'] = df_academics['Date'] + ' (' + df_academics['Day'].str[:3] + ')'


    # Create two columns for the plots
    col1, middle_space, col2 = st.columns([1, 0.1, 1])  # Adjust the middle space as needed (0.1 for 10% width)

    # Plot the pie chart in the first column with some margin/padding
    with col1:
        # Compute the total hours spent for each activity under the Personal category
        total_duration = df_academics.groupby('Activity')['Duration (hrs)'].sum().reset_index()

        # Create the pie chart
        pie_fig = px.pie(total_duration, names='Activity', values='Duration (hrs)', title='Total Hours for Each Academic Activity')
        st.plotly_chart(pie_fig)

    # Add a little space in the middle
    with middle_space:
        pass

    # Plot the bar chart in the second column with some margin/padding
    with col2:
        # Getting the frequency of each activity within the 'Personal' category
        grouped = df_academics['Activity'].value_counts().reset_index()
        grouped.columns = ['Activity', 'Frequency']

        # Plotting the data using plotly
        fig = px.bar(grouped, x='Activity', y='Frequency', title='Frequency of Each Academic Activity.')

        st.plotly_chart(fig)


    
    fig = px.bar(df_academics, x='Date & Day', y='Duration (hrs)', color='Activity', title='Duration of Academic Activities over Days')
    st.plotly_chart(fig)


elif selected_menu == "Work / Business":
    st.header("Work/Business Activities")
    st.write("Here's a closer look at how I allocate my hours in the realm of work and business. From essential meetings to task execution, and the occasional waiting periods, this breakdown offers an intimate glance into the rhythm of my daily business endeavors.")
    st.write("\n")

    # Filter by category
    df_work = df[df["Category"] == "Work / Business"]

    # Compute the total hours spent for each activity under the Personal category
    total_duration = df_work.groupby('Activity')['Duration (hrs)'].sum()
    unique_days = df_work.groupby('Activity')['Date'].nunique()
    avg_per_day = (total_duration / unique_days).round(2)

    # Collecting metrics HTML in this string
    metrics_html = ""

    for activity, avg_duration in avg_per_day.items():
        metrics_html += f"""
        <div class="customMetric">
            <h5>{activity}</h5>
            <h2>{avg_duration}</h2>
            <label>Average Hours Spent</label>
        </div>
        """

    # Render the metrics
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.markdown('<hr/>', unsafe_allow_html=True);


    # Create two columns for the plots
    col1, middle_space, col2 = st.columns([1, 0.1, 1])  # Adjust the middle space as needed (0.1 for 10% width)

    # Plot the pie chart in the first column with some margin/padding
    with col1:
        # Compute the total hours spent for each activity under the Personal category
        total_duration = df_work.groupby('Activity')['Duration (hrs)'].sum().reset_index()

        # Create the pie chart
        pie_fig = px.pie(total_duration, names='Activity', values='Duration (hrs)', title='Total Hours for Each Work Activity')
        st.plotly_chart(pie_fig)

    # Add a little space in the middle
    with middle_space:
        pass

    # Plot the bar chart in the second column with some margin/padding
    with col2:
        # Getting the frequency of each activity
        grouped = df_work['Activity'].value_counts().reset_index()
        grouped.columns = ['Activity', 'Frequency']

        # Plotting the data using plotly
        fig = px.bar(grouped, x='Activity', y='Frequency', title='Frequency of Each Work/Business Activity')

        st.plotly_chart(fig)

    
    df_work['Date & Day'] = df_work['Date'] + ' (' + df_work['Day'].str[:3] + ')'
    

    fig = px.bar(df_work, x='Date & Day', y='Duration (hrs)', color='Activity', title='Duration of Work/Business Activities over Days')
    st.plotly_chart(fig)


elif selected_menu == "Entertainment":
    st.header("Entertainment Activities")
    st.write("The chart below offers a glimpse into how I spend my free moments, primarily oscillating between the realms of social media and gaming. While the allure of digital connections is unmistakable, I also find solace in the immersive worlds of various games. Here's a visual representation of my daily relaxation rituals.")
    st.write("\n")


    # Filter by category
    df_entertainment = df[df["Category"] == "Entertainment"]

    # Compute the total hours spent for each activity under the Personal category
    total_duration = df_entertainment.groupby('Activity')['Duration (hrs)'].sum()
    unique_days = df_entertainment.groupby('Activity')['Date'].nunique()
    avg_per_day = (total_duration / unique_days).round(2)

    # Collecting metrics HTML in this string
    metrics_html = ""

    metrics_html += f"""
        <div class="customMetric" style="opacity: 0;">
     
        </div>
        """

    for activity, avg_duration in avg_per_day.items():
        metrics_html += f"""
        <div class="customMetric">
            <h5>{activity}</h5>
            <h2>{avg_duration}</h2>
            <label>Average Hours Spent</label>
        </div>
        """

    # Render the metrics
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.markdown('<hr/>', unsafe_allow_html=True);



    # Create two columns for the plots
    col1, middle_space, col2 = st.columns([1, 0.1, 1])  # Adjust the middle space as needed (0.1 for 10% width)

    # Plot the pie chart in the first column with some margin/padding
    with col1:
       # Compute the total hours spent for each activity under the Personal category
        total_duration = df_entertainment.groupby('Activity')['Duration (hrs)'].sum().reset_index()

        # Create the pie chart
        pie_fig = px.pie(total_duration, names='Activity', values='Duration (hrs)', title='Total Hours for Each Entertainment Activity')
        st.plotly_chart(pie_fig)

    # Add a little space in the middle
    with middle_space:
        pass

    # Plot the bar chart in the second column with some margin/padding
    with col2:
        # Getting the frequency of each activity
        grouped = df_entertainment['Activity'].value_counts().reset_index()
        grouped.columns = ['Activity', 'Frequency']

        # Plotting the data using plotly
        fig = px.bar(grouped, x='Activity', y='Frequency', title='Frequency of Each Entertainment Activity')

        st.plotly_chart(fig)
        
    df_entertainment['Date & Day'] = df_entertainment['Date'] + ' (' + df_entertainment['Day'].str[:3] + ')'
    
    fig = px.bar(df_entertainment, x='Date & Day', y='Duration (hrs)', color='Activity', title='Duration of Entertainment Activities over Days')
    st.plotly_chart(fig)
    st.markdown('<hr/>', unsafe_allow_html=True);
