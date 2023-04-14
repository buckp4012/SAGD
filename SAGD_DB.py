import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import dash
from dash import dash_table
from dash import dcc
from dash import html
import math
from dash import dash_table
import plotly.graph_objs as go




#Load data from CSV file or pandas dataframe
data = pd.read_excel(r'C:\Users\H277910\SAGD Dashboard V2.xlsx')

app = dash.Dash(__name__)
server = app.server






#Define the layout of the app
app.layout = html.Div(
    style={'backgroundColor': '#b22222', 'height': '120px'},
    children=[

        html.Hr(),
        html.H1('Summit ESP- A Halliburton Service: SAGD Dashboard',
                style={"text-align": "center", "font-size": "3rem"}),

        html.Div(style={'height': '40px', "backgroundColor": "white"}),

        dcc.Dropdown(
            id='customer-dropdown',
            options=[{'label': 'All Customers', 'value': 'all'},
                     {'label': 'CENOVUS', 'value': 'CENOVUS'},
                     {'label': 'CNRL', 'value': 'CNRL'},
                     {'label': 'CONOCO', 'value': 'CONOCO'},
                     {'label': 'CNOOC', 'value': 'CNOOC'},
                     {'label': 'HARVEST', 'value': 'HARVEST'},
                     {'label': 'STRATHCONA', 'value': 'STRATHCONA'},
                     {'label': 'SUNCOR', 'value': 'SUNCOR'}
                     ],
            value='all',
            #placeholder='Select a customer...'
        ),
        html.Div(style={'height': '30px', "backgroundColor": "white"}),
        dcc.Graph(id='data-table'),
        dcc.Dropdown(
            id='seal-dropdown',
            options=[
                {'label': 'All Versions', 'value': 'all'},
                {'label': 'Version 1', 'value': 'Version 1'},
                {'label': 'Version 2', 'value': 'Version 2'},
                {'label': 'Version 2.5', 'value': 'Version 2.5'},
                {'label': 'Version 3.0', 'value': 'Version 3.0'},
                {'label': 'Version 3.0- New screen', 'value': 'Version 3.0- New screen'}
            ],
            value='all'
        ),
        
        dcc.Graph(id='failure-points-pie-chart'),
        dcc.Graph(id='reason-for-pull-pie-chart'),
        dcc.Graph(id='normal-dist-plot'),
        dcc.Graph(id='survive-plot'),
        dcc.Graph(id='reliability-plot'),
        html.Footer(children=[html.P("Summit ESP - Global Technical Service; Created by Buck Pettit - 2023",
                                     style={"font-size": "small"})])
    ])


# Define the callbacks of the app
@app.callback(
    [dash.dependencies.Output('data-table', 'figure'),
        dash.dependencies.Output('failure-points-pie-chart', 'figure'),
     dash.dependencies.Output('reason-for-pull-pie-chart', 'figure')],
    dash.dependencies.Output('normal-dist-plot', 'figure'),
    dash.dependencies.Output('survive-plot', 'figure'),
    dash.dependencies.Output('reliability-plot', 'figure'),
    [dash.dependencies.Input('seal-dropdown', 'value'),
     dash.dependencies.Input('customer-dropdown', 'value')]
)
def update_pie_charts(selected_seal_version, selected_customer):
    # Filter the data by the selected customer and SEAL version if needed
    if selected_customer != 'all':
        filtered_data = data[data['CUSTOMER'] == selected_customer]
        runtime_col = filtered_data['RUNTIME (Excel Calculated)'].dropna()
        avg = round(runtime_col.mean())
        max_run = runtime_col.max()
        install_total = runtime_col.count()
        count_total = len(filtered_data[filtered_data['R/NR/Waiting'] == 'NR'])
        count_ver1 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 1') & (filtered_data['R/NR/Waiting'] == 'NR')])
        count_ver2 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 2') & (filtered_data['R/NR/Waiting'] == 'NR')])
        count_ver2_5 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 2.5') & (filtered_data['R/NR/Waiting'] == 'NR')])
        count_ver3 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 3.0') & (filtered_data['R/NR/Waiting'] == 'NR')])
        count_ver3_ns = len(filtered_data[(filtered_data['SEAL'] == 'Version 3.0- New screen') & (
                filtered_data['R/NR/Waiting'] == 'NR')])
        Run_total = len(filtered_data[filtered_data['R/NR/Waiting'] == 'R'])
        Run_ver1 = len(filtered_data[(filtered_data['SEAL'] == 'Version 1') & (filtered_data['R/NR/Waiting'] == 'R')])
        Run_ver2 = len(filtered_data[(filtered_data['SEAL'] == 'Version 2') & (filtered_data['R/NR/Waiting'] == 'R')])
        Run_ver2_5 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 2.5') & (filtered_data['R/NR/Waiting'] == 'R')])
        Run_ver3 = len(filtered_data[(filtered_data['SEAL'] == 'Version 3.0') & (filtered_data['R/NR/Waiting'] == 'R')])
        Run_ver3_ns = len(filtered_data[(filtered_data['SEAL'] == 'Version 3.0- New screen') & (
                filtered_data['R/NR/Waiting'] == 'R')])

        days_total = filtered_data[filtered_data['R/NR/Waiting'] == 'NR']['RUNTIME (Excel Calculated)'].sum()
        days_ver1 = filtered_data[(filtered_data['SEAL'] == 'Version 1') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()
        days_ver2 = filtered_data[(filtered_data['SEAL'] == 'Version 2') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()
        days_ver2_5 = filtered_data[(filtered_data['SEAL'] == 'Version 2.5') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()
        days_ver3 = filtered_data[(filtered_data['SEAL'] == 'Version 3.0') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()
        days_ver3_ns = \
        filtered_data[(filtered_data['SEAL'] == 'Version 3.0- New screen') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()

        subset1 = filtered_data.loc[filtered_data['SEAL'] == 'Version 1']
        avg1 = round(subset1['RUNTIME (Excel Calculated)'].mean())
        max_run1 = subset1['RUNTIME (Excel Calculated)'].max()
        install1 = subset1['RUNTIME (Excel Calculated)'].count()
        subset2 = filtered_data.loc[filtered_data['SEAL'] == 'Version 2']
        avg2 = subset2['RUNTIME (Excel Calculated)'].mean()
        max_run2 = subset2['RUNTIME (Excel Calculated)'].max()
        install2 = subset2['RUNTIME (Excel Calculated)'].count()
        subset2_5 = filtered_data.loc[filtered_data['SEAL'] == 'Version 2.5']
        avg2_5 = subset2_5['RUNTIME (Excel Calculated)'].mean()
        max_run2_5 = (subset2_5['RUNTIME (Excel Calculated)'].max())
        install2_5 = subset2_5['RUNTIME (Excel Calculated)'].count()
        subset3 = filtered_data.loc[filtered_data['SEAL'] == 'Version 3.0']
        avg3 = subset3['RUNTIME (Excel Calculated)'].mean()
        max_run3 = subset3['RUNTIME (Excel Calculated)'].max()
        install3 = subset3['RUNTIME (Excel Calculated)'].count()
        subset3_ns = filtered_data.loc[filtered_data['SEAL'] == 'Version 3.0- New screen']
        avg3_ns = subset3_ns['RUNTIME (Excel Calculated)'].mean()
        max_run3_ns = subset3_ns['RUNTIME (Excel Calculated)'].max()
        install3_ns = subset3_ns['RUNTIME (Excel Calculated)'].count()
        
        
        
        if selected_seal_version != 'all':
            filtered_data = filtered_data[filtered_data['SEAL'] == selected_seal_version]
    else:
        filtered_data = data
        runtime_col = filtered_data['RUNTIME (Excel Calculated)'].dropna()
        avg = round(runtime_col.mean())
        max_run = runtime_col.max()
        install_total = runtime_col.count()
        count_total = len(filtered_data[filtered_data['R/NR/Waiting'] == 'NR'])
        count_ver1 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 1') & (filtered_data['R/NR/Waiting'] == 'NR')])
        count_ver2 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 2') & (filtered_data['R/NR/Waiting'] == 'NR')])
        count_ver2_5 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 2.5') & (filtered_data['R/NR/Waiting'] == 'NR')])
        count_ver3 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 3.0') & (filtered_data['R/NR/Waiting'] == 'NR')])
        count_ver3_ns = len(filtered_data[(filtered_data['SEAL'] == 'Version 3.0- New screen') & (
                filtered_data['R/NR/Waiting'] == 'NR')])
        Run_total = len(filtered_data[filtered_data['R/NR/Waiting'] == 'R'])
        Run_ver1 = len(filtered_data[(filtered_data['SEAL'] == 'Version 1') & (filtered_data['R/NR/Waiting'] == 'R')])
        Run_ver2 = len(filtered_data[(filtered_data['SEAL'] == 'Version 2') & (filtered_data['R/NR/Waiting'] == 'R')])
        Run_ver2_5 = len(
            filtered_data[(filtered_data['SEAL'] == 'Version 2.5') & (filtered_data['R/NR/Waiting'] == 'R')])
        Run_ver3 = len(filtered_data[(filtered_data['SEAL'] == 'Version 3.0') & (filtered_data['R/NR/Waiting'] == 'R')])
        Run_ver3_ns = len(filtered_data[(filtered_data['SEAL'] == 'Version 3.0- New screen') & (
                filtered_data['R/NR/Waiting'] == 'R')])

        days_total = filtered_data[filtered_data['R/NR/Waiting'] == 'NR']['RUNTIME (Excel Calculated)'].sum()
        days_ver1 = filtered_data[(filtered_data['SEAL'] == 'Version 1') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()
        days_ver2 = filtered_data[(filtered_data['SEAL'] == 'Version 2') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()
        days_ver2_5 = filtered_data[(filtered_data['SEAL'] == 'Version 2.5') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()
        days_ver3 = filtered_data[(filtered_data['SEAL'] == 'Version 3.0') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()
        days_ver3_ns = filtered_data[(filtered_data['SEAL'] == 'Version 3.0- New screen') & (filtered_data['R/NR/Waiting'] == 'NR')][
            'RUNTIME (Excel Calculated)'].sum()


        subset1 = filtered_data.loc[filtered_data['SEAL'] == 'Version 1']
        avg1 = subset1['RUNTIME (Excel Calculated)'].mean()
        max_run1 = subset1['RUNTIME (Excel Calculated)'].max()
        install1 = subset1['RUNTIME (Excel Calculated)'].count()
        subset2 = filtered_data.loc[filtered_data['SEAL'] == 'Version 2']
        avg2 = subset2['RUNTIME (Excel Calculated)'].mean()
        max_run2 = subset2['RUNTIME (Excel Calculated)'].max()
        install2 = subset2['RUNTIME (Excel Calculated)'].count()
        subset2_5 = filtered_data.loc[filtered_data['SEAL'] == 'Version 2.5']
        avg2_5 = subset2_5['RUNTIME (Excel Calculated)'].mean()
        max_run2_5 = subset2_5['RUNTIME (Excel Calculated)'].max()
        install2_5 = subset2_5['RUNTIME (Excel Calculated)'].count()
        subset3 = filtered_data.loc[filtered_data['SEAL'] == 'Version 3.0']
        avg3 = subset3['RUNTIME (Excel Calculated)'].mean()
        max_run3 = subset3['RUNTIME (Excel Calculated)'].max()
        install3 = subset3['RUNTIME (Excel Calculated)'].count()
        subset3_ns = filtered_data.loc[filtered_data['SEAL'] == 'Version 3.0- New screen']
        avg3_ns = subset3_ns['RUNTIME (Excel Calculated)'].mean()
        max_run3_ns = subset3_ns['RUNTIME (Excel Calculated)'].max()
        install3_ns = subset3_ns['RUNTIME (Excel Calculated)'].count()
        if selected_seal_version != 'all':
            filtered_data = data[data['SEAL'] == selected_seal_version]
        else:
            filtered_data = data

    # Generate the pie charts
    failure_points = filtered_data['Failure Points'].value_counts()
    reason_for_pull = filtered_data['Reason for Pull'].value_counts()
    Runtime_rows = filtered_data[filtered_data['R/NR/Waiting'] == 'NR']
    drop_blanks = filtered_data['RUNTIME (Excel Calculated)'].dropna()
    Sum_run=sum(drop_blanks)
    Runtime = Runtime_rows['RUNTIME (Excel Calculated)'].tolist()
    sorted_runtime=sorted(Runtime)

    if days_total==0 or count_total==0:
        mttf_total="N/A"
    else:
        mttf_total = round(days_total / count_total)
    if days_ver1==0 or count_ver1==0:
        mttf_ver1="N/A"
    else:
        mttf_ver1 = round(days_ver1 / count_ver1)
    if days_ver2==0 or count_ver2==0:
        mttf_ver2="N/A"
    else:
        mttf_ver2 = round(days_ver2 / count_ver2)
    if days_ver2_5==0 or count_ver2_5==0:
        mttf_ver2_5="N/A"
    else:
        mttf_ver2_5 = round(days_ver2_5 / count_ver2_5)
    if days_ver3==0 or count_ver3==0:
        mttf_ver3="N/A"
    else:
        mttf_ver3 = round(days_ver3 / count_ver3)
    if days_ver3_ns == 0 or count_ver3_ns == 0:
        mttf_ver3_ns = "N/A"
    else:
        mttf_ver3_ns = round(days_ver3_ns / count_ver3_ns)
    if install_total==0:
        avg_new="N/A"
        max_run="N/A"
        Run_total="N/A"
    else:
        avg_new=round(avg)
    if install1==0:
        avg1_new="N/A"
        max_run1="N/A"
        Run_ver1="N/A"
    else:
        avg1_new=round(avg1)
    if install2==0:
        avg2_new="N/A"
        max_run2="N/A"
        Run_ver2="N/A"
    else:
        avg2_new=round(avg2)
    if install2_5==0:
        avg2_5_new="N/A"
        max_run2_5="N/A"
        Run_ver2_5="N/A"
    else:
        avg2_5_new=round(avg2_5)
    if install3==0:
        avg3_new="N/A"
        max_run3="N/A"
        Run_ver3="N/A"
    else:
        avg3_new=round(avg3)
    if install3_ns==0:
        avg3_ns_new="N/A"
        max_run3_ns="N/A"
        Run_ver3_ns="N/A"
    else:
        avg3_ns_new=round(avg3_ns)
    
    table_data = pd.DataFrame(
        {'Seal Version': ['All Versions', 'Version 1', 'Version 2', 'Version 2.5', 'Version 3',
                          'Version 3.0- New screen'],
         'Average Runtime': [avg_new, avg1_new, avg2_new, avg2_5_new, avg3_new, avg3_ns_new],
         'Installs': [install_total, install1, install2, install2_5, install3, install3_ns],
         'Actively Running': [Run_total, Run_ver1, Run_ver2, Run_ver2_5, Run_ver3, Run_ver3_ns],
         'MTTF (days)': [mttf_total, mttf_ver1, mttf_ver2, mttf_ver2_5, mttf_ver3, mttf_ver3_ns],
         'Max Runtime': [max_run, max_run1, max_run2, max_run2_5, max_run3, max_run3_ns]
         })
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=list(table_data.columns),
                    align='center',
                    font=dict(size=25),
                    height=50),

        cells=dict(values=[table_data['Seal Version'], table_data['Average Runtime'],
                           table_data['Installs'], table_data['Actively Running'], table_data['MTTF (days)'],
                           table_data['Max Runtime']],
                   align='center',
                   font=dict(size=20),
                   height=40))])

    table_fig.update_layout(height=500)
    mean=np.mean(sorted_runtime)
    std_dev=np.std(sorted_runtime)
    dist=norm(loc=mean, scale= std_dev)
    pdf_values=[]
    for value in sorted_runtime:
        pdf_values.append(dist.pdf(value))
    max_1=np.max(pdf_values)

    failure_points_fig = px.pie(
        names=failure_points.index,
        values=failure_points.values,
        title=f'Failure Points for All Versions (Total: {sum(failure_points)})'
    )
    reason_for_pull_fig = px.pie(
        names=reason_for_pull.index,
        values=reason_for_pull.values,
        title=f'Reason for Pull for All Versions (Total: {sum(failure_points)})'
    )
    normal_dist_fig = px.scatter(x=sorted_runtime, y=pdf_values, title=f'Normal Distribution of Runtime (Total: {sum(failure_points)})'
    )
    normal_dist_fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Normal Distribution",
        shapes=[
            dict(
                type='line',
                x0=mean,
                y0=0,
                x1=mean,
                y1=max_1,
                line=dict(color='red', width=2),
            )
        ],
        annotations=[
            dict(
                x=mean,
                y=max_1,
                xref="x",
                yref="y",
                text="mean =" + str(round(mean)) + " days",

            )
        ], )
    failure_count = []
    prob_array = []

    for i in range(len(sorted_runtime)):
        failure_count.append(i + 1)
    for i in range(len(sorted_runtime)):
        results = 1 - (failure_count[i] / len(failure_count))
        prob_array.append(results)
    survivability_fig = px.line(
        x=sorted_runtime,
        y=prob_array,
        title=f'Survivability Curve for {selected_seal_version} (Total: {sum(failure_points)})')
    survivability_fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Survival Probability",
    )
    Total_failures=np.max(failure_count)
    mttf=(Sum_run/Total_failures)
    
    reliability = []
    for i in range(len(sorted_runtime)):
        reliability.append(math.exp((-1 / mttf) * sorted_runtime[i]))
    reliability_fig = px.line(
        x=sorted_runtime,
        y=reliability,
        title=f'Reliability Curve for {selected_seal_version} (Total: {sum(failure_points)})'
    )
    reliability_fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Overall Reliability",)

    return table_fig, failure_points_fig, reason_for_pull_fig, normal_dist_fig,survivability_fig, reliability_fig
    


if __name__ == '__main__':
    app.run_server(debug=True)
