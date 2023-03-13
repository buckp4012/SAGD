import pandas as pd
import numpy as np
from numpy import average
from scipy.stats import norm
import plotly.express as px
import dash
from dash import dash_table
from dash import dcc
from dash import html
import base64



# Load data from CSV file or pandas dataframe
data = pd.read_excel('SAGD Dashboard V2.xlsx')

with open('Summit_logo.png') as f:
    image_data = f.read()

encoded_image=base64.b64encode(image_data).decode('utf-8')

nr_rows = data.loc[data['R/NR/Waiting'] == 'NR']

# extract the values from the "Runtime" column of the filtered rows
runtime_values = nr_rows['RUNTIME (Excel Calculated)'].values
sorted_runtime = sorted(runtime_values)
mean = np.mean(sorted_runtime)
std_dev = np.std(sorted_runtime)
dist = norm(loc=mean, scale=std_dev)
pdf_values = []
for value in sorted_runtime:
    pdf_values.append(dist.pdf(value))
max = np.max(pdf_values)
failure_count=[]
prob_array=[]
for i in range(len(sorted_runtime)):
    failure_count.append(i+1)
for i in range(len(sorted_runtime)):
    results=1-(failure_count[i]/len(failure_count))
    prob_array.append(results)
    

#ver1
nr_rows1 = data[(data['R/NR/Waiting'] == 'NR') & (data['SEAL'] == 'ver 1')]

# extract the values from the "Runtime" column of the filtered rows
runtime_values1 = nr_rows1['RUNTIME (Excel Calculated)'].values
sorted_runtime1 = sorted(runtime_values1)
mean1 = np.mean(sorted_runtime1)
std_dev1 = np.std(sorted_runtime1)
dist1 = norm(loc=mean1, scale=std_dev1)
pdf_values1 = []
for value in sorted_runtime1:
    pdf_values1.append(dist1.pdf(value))
max1 = np.max(pdf_values1)
failure_count1=[]
prob_array1=[]
for i in range(len(sorted_runtime1)):
    failure_count1.append(i+1)
for i in range(len(sorted_runtime1)):
    results=1-(failure_count1[i]/len(failure_count1))
    prob_array1.append(results)



#ver 2
nr_rows2 = data[(data['R/NR/Waiting'] == 'NR') & (data['SEAL'] == 'ver 2')]

# extract the values from the "Runtime" column of the filtered rows
runtime_values2 = nr_rows2['RUNTIME (Excel Calculated)'].values
sorted_runtime2 = sorted(runtime_values2)
mean2 = np.mean(sorted_runtime2)
std_dev2 = np.std(sorted_runtime2)
dist2 = norm(loc=mean2, scale=std_dev2)
pdf_values2 = []
for value in sorted_runtime2:
    pdf_values2.append(dist2.pdf(value))
max2 = np.max(pdf_values2)
failure_count2=[]
prob_array2=[]
for i in range(len(sorted_runtime2)):
    failure_count2.append(i+1)
for i in range(len(sorted_runtime2)):
    results=1-(failure_count2[i]/len(failure_count2))
    prob_array2.append(results)


#ver 2.5
nr_rows2_5 = data[(data['R/NR/Waiting'] == 'NR') & (data['SEAL'] == 'ver 2.5')]

runtime_values2_5 = nr_rows2_5['RUNTIME (Excel Calculated)'].values
sorted_runtime2_5 = sorted(runtime_values2_5)
mean2_5 = np.mean(sorted_runtime2_5)
std_dev2_5 = np.std(sorted_runtime2_5)
dist2_5 = norm(loc=mean2_5, scale=std_dev2_5)
pdf_values2_5 = []
for value in sorted_runtime2_5:
    pdf_values2_5.append(dist2_5.pdf(value))
max2_5 = np.max(pdf_values2_5)
failure_count2_5=[]
prob_array2_5=[]
for i in range(len(sorted_runtime2_5)):
    failure_count2_5.append(i+1)
for i in range(len(sorted_runtime2_5)):
    results=1-(failure_count2_5[i]/len(failure_count2_5))
    prob_array2_5.append(results)

#ver 3
nr_rows3 = data[(data['R/NR/Waiting'] == 'NR') & (data['SEAL'] == 'ver 3.0')]

# extract the values from the "Runtime" column of the filtered rows
runtime_values3 = nr_rows3['RUNTIME (Excel Calculated)'].values
sorted_runtime3 = sorted(runtime_values3)
mean3 = np.mean(sorted_runtime3)
std_dev3 = np.std(sorted_runtime3)
dist3 = norm(loc=mean3, scale=std_dev3)
pdf_values3 = []
for value in sorted_runtime3:
    pdf_values3.append(dist3.pdf(value))
max3 = np.max(pdf_values3)
failure_count3=[]
prob_array3=[]
for i in range(len(sorted_runtime3)):
    failure_count3.append(i+1)
for i in range(len(sorted_runtime3)):
    results=1-(failure_count2_5[i]/len(failure_count3))
    prob_array3.append(results)


#ver 3.5
nr_rows3_ns = data[(data['R/NR/Waiting'] == 'NR') & (data['SEAL'] == 'ver 3.0-New screen')]

# extract the values from the "Runtime" column of the filtered rows
runtime_values3_ns = nr_rows3_ns['RUNTIME (Excel Calculated)'].values
if len(runtime_values3_ns)==0:
    sorted_runtime3_ns=0
    mean3_ns=0
    std_dev3_ns=0
    dist3_ns=0
    pdf_values3_ns=0
    max3_ns=0
    failure_count3_ns = 0
    prob_array3_ns = 0
else:

    sorted_runtime3_ns = sorted(runtime_values3_ns)
    mean3_ns = np.mean(sorted_runtime3_ns)
    std_dev3_ns = np.std(sorted_runtime3_ns)
    dist3_ns = norm(loc=mean3_ns, scale=std_dev3_ns)
    pdf_values3_ns = []
    for value in sorted_runtime3_ns:
        pdf_values3_ns.append(dist3_ns.pdf(value))
    max3_ns = np.max(pdf_values3_ns)
    failure_count3_ns = []
    prob_array3_ns = []
    for i in range(len(sorted_runtime3_ns)):
        failure_count3_ns.append(i + 1)
    for i in range(len(sorted_runtime3_ns)):
        results = 1 - (failure_count3_ns[i] / len(failure_count3_ns))
        prob_array3_ns.append(results)

count_total=len(data[data['R/NR/Waiting']=='NR'])
count_ver1=len(data[(data['SEAL']=='ver 1') & (data['R/NR/Waiting']=='NR')])
count_ver2=len(data[(data['SEAL']=='ver 2') & (data['R/NR/Waiting']=='NR')])
count_ver2_5=len(data[(data['SEAL']=='ver 2.5') & (data['R/NR/Waiting']=='NR')])
count_ver3=len(data[(data['SEAL']=='ver 3.0') & (data['R/NR/Waiting']=='NR')])
count_ver3_ns=len(data[(data['SEAL']=='ver 3.0- New screen') & (data['R/NR/Waiting']=='NR')])


days_total=data[data['R/NR/Waiting']=='NR']['RUNTIME (Excel Calculated)'].sum()
days_ver1=data[(data['SEAL']=='ver 1') & (data['R/NR/Waiting']=='NR')]['RUNTIME (Excel Calculated)'].sum()
days_ver2=data[(data['SEAL']=='ver 2') & (data['R/NR/Waiting']=='NR')]['RUNTIME (Excel Calculated)'].sum()
days_ver2_5=data[(data['SEAL']=='ver 2.5') & (data['R/NR/Waiting']=='NR')]['RUNTIME (Excel Calculated)'].sum()
days_ver3=data[(data['SEAL']=='ver 3.0') & (data['R/NR/Waiting']=='NR')]['RUNTIME (Excel Calculated)'].sum()
days_ver3_ns=data[(data['SEAL']=='ver 3.0- New screen') & (data['R/NR/Waiting']=='NR')]['RUNTIME (Excel Calculated)'].sum()


mttf_total=round(days_total/count_total)
mttf_ver1=round(days_ver1/count_ver1)
mttf_ver2=round(days_ver2/count_ver2)
mttf_ver2_5=round(days_ver2_5/count_ver2_5)
mttf_ver3=round(days_ver3/count_ver3)
if count_ver3_ns==0:
    mttf_ver3_ns=0
else:
    mttf_ver3_ns=(days_ver3_ns/count_ver3_ns)

install_total=len(data['W/L']=='w')
install1=len(data[(data['SEAL']=='ver 1') & (data['W/L']=='w')])
install2=len(data[(data['SEAL']=='ver 2')& (data['W/L']=='w')])
install2_5=len(data[(data['SEAL']=='ver 2.5')& (data['W/L']=='w')])
install3=len(data[(data['SEAL']=='ver 3.0')& (data['W/L']=='w')])
install3_ns=len(data[(data['SEAL']=='ver 3.0- New screen')& (data['W/L']=='w')])

runtime_col=data['RUNTIME (Excel Calculated)'].dropna()

avg=round(runtime_col.mean())
avg1=round(data[(data['SEAL']=='ver 1')]['RUNTIME (Excel Calculated)'].mean())
avg2=round(data[(data['SEAL']=='ver 2')]['RUNTIME (Excel Calculated)'].mean())
avg2_5=round(data[(data['SEAL']=='ver 2.5')]['RUNTIME (Excel Calculated)'].mean())
avg3=round(data[(data['SEAL']=='ver 3.0')]['RUNTIME (Excel Calculated)'].mean())
avg3_ns=round(data[(data['SEAL']=='ver 3.0- New screen')]['RUNTIME (Excel Calculated)'].mean())



table_data=pd.DataFrame(
    {'Seal Version': ['All Versions','ver 1', 'ver 2', 'ver 2.5', 'ver 3','ver 3.0-New screen'],
     'MTTF (days)': [mttf_total,mttf_ver1, mttf_ver2, mttf_ver2_5, mttf_ver3, mttf_ver3_ns],
     'Installs': [install_total, install1,install2,install2_5,install3,install3_ns],
     'Average Runtime': [avg,avg1,avg2,avg2_5,avg3,avg3_ns]})



# Create a Dash app
app = dash.Dash(__name__)
server=app.server

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1('Summit ESP- A Halliburton Service: SAGD Dashboard'),
    #html.Img(src=r'C:\Users\H277910\OneDrive - Halliburton\Pictures\Summit_logo.png'),
    dash_table.DataTable(
        id='table',
        columns=[{"name":i, "id":i} for i in table_data.columns],
        data=table_data.to_dict('records'),
        style_table={
            'height':'300px',
            'overflowY': 'auto',
            ',axWidth': '50px',
            'textAlign': 'center'
        },
        style_cell={'textAlign': 'center',
                    'maxWidth': '25px'}
    ),
    dcc.Dropdown(
        id='seal-dropdown',
        options=[
            {'label': 'All Versions', 'value': 'all'},
            {'label': 'ver 1', 'value': 'ver 1'},
            {'label': 'ver 2', 'value': 'ver 2'},
            {'label': 'ver 2.5', 'value': 'ver 2.5'},
            {'label': 'ver 3.0', 'value': 'ver 3.0'},
            {'label': 'ver 3.0-New screen', 'value': 'ver 3.0-New screen'}
        ],
        value='all'
    ),
    dcc.Graph(id='failure-points-pie-chart'),
    dcc.Graph(id='reason-for-pull-pie-chart'),
    dcc.Graph(id='normal-dist-plot'),
    dcc.Graph(id='survive-plot')
])

# Define the callbacks of the app
@app.callback(
    [dash.dependencies.Output('failure-points-pie-chart', 'figure'),
     dash.dependencies.Output('reason-for-pull-pie-chart', 'figure')],
    dash.dependencies.Output('normal-dist-plot', 'figure'),
    dash.dependencies.Output('survive-plot', 'figure'),
    [dash.dependencies.Input('seal-dropdown', 'value')]
)

def update_pie_charts(selected_seal_version):
    # Filter the data by the selected SEAL version if needed
    if selected_seal_version != 'all':
        filtered_data = data[data['SEAL'] == selected_seal_version]
    else:
        filtered_data = data

    # Generate the pie charts
    failure_points = data['Failure Points'].value_counts()
    reason_for_pull = data['Reason for Pull'].value_counts()

    failure_points_fig = px.pie(
        names=failure_points.index,
        values=failure_points.values,
        title=f'Failure Points for All Versions (Total: {count_total})'
    )
    reason_for_pull_fig = px.pie(
        names=reason_for_pull.index,
        values=reason_for_pull.values,
        title=f'Reason for Pull for All Versions (Total: {count_total})'
    )
    normal_dist_fig = px.scatter(
        x=sorted_runtime,
        y=pdf_values,
        title=f'Normal Distribution (Total: {count_total})'
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
                y1=max,
                line=dict(color='red', width=2),
            )],
        annotations=[
            dict(
                x=mean,
                y=max,
                xref="x",
                yref="y",
                text="mean ="+str(round(mean))+" days",

            )
        ]
    )
    survivability_fig = px.line(
        x=sorted_runtime,
        y=prob_array,
        title=f'Survivability Curve (Total: {count_total})'
    )
    survivability_fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Survival Probability",
        )
    # Generate the filtered pie charts if needed
    if selected_seal_version != 'all':
        filtered_failure_points = filtered_data['Failure Points'].value_counts()
        filtered_reason_for_pull = filtered_data['Reason for Pull'].value_counts()

        filtered_failure_points_fig = px.pie(
            names=filtered_failure_points.index,
            values=filtered_failure_points.values,
            title=f'Failure Points for {selected_seal_version} (Total: {sum(filtered_failure_points)})'
        )
        filtered_reason_for_pull_fig = px.pie(
            names=filtered_reason_for_pull.index,
            values=filtered_reason_for_pull.values,
            title=f'Reason for Pull for {selected_seal_version} (Total: {sum(filtered_failure_points)})'
        )
        if selected_seal_version == "ver 1":
            filtered_normal = px.scatter(
                x=sorted_runtime1,
                y=pdf_values1,
                title=f'Normal Distribution for {selected_seal_version} (Total: {sum(filtered_failure_points)})',
            )
            filtered_normal.update_layout(
                xaxis_title="Days",
                yaxis_title="Normal Distribution",
                shapes=[
                    dict(
                        type='line',
                        x0=mean1,
                        y0=0,
                        x1=mean1,
                        y1=max1,
                        line=dict(color='red',width=2),
                    )
                    ],
                annotations = [
                dict(
                    x=mean1,
                    y=max1,
                    xref="x",
                    yref="y",
                    text="mean =" + str(round(mean1)) + " days",

                )
            ],)
            filtered_survivability_fig = px.line(
                x=sorted_runtime1,
                y=prob_array1,
                title=f'Survivability Curve for {selected_seal_version} (Total: {sum(filtered_failure_points)})'
            )
            filtered_survivability_fig.update_layout(
                xaxis_title="Days",
                yaxis_title="Survival Probability",
            )
        if selected_seal_version == "ver 2":
            filtered_normal = px.scatter(
                x=sorted_runtime2,
                y=pdf_values2,
                title=f'Normal Distribution for {selected_seal_version} (Total: {sum(filtered_failure_points)})',
            )
            filtered_normal.update_layout(
                xaxis_title="Days",
                yaxis_title="Normal Distribution",
                shapes=[
                    dict(
                        type='line',
                        x0=mean2,
                        y0=0,
                        x1=mean2,
                        y1=max2,
                        line=dict(color='red', width=2),
                    )
                ],
                annotations=[
                    dict(
                        x=mean2,
                        y=max2,
                        xref="x",
                        yref="y",
                        text="mean =" + str(round(mean2)) + " days",

                    )
                ], )
            filtered_survivability_fig = px.line(
                x=sorted_runtime2,
                y=prob_array2,
                title=f'Survivability Curve for {selected_seal_version} (Total: {sum(filtered_failure_points)})'
            )
            filtered_survivability_fig.update_layout(
                xaxis_title="Days",
                yaxis_title="Survival Probability",
            )
        if selected_seal_version == "ver 2.5":
            filtered_normal = px.scatter(
                x=sorted_runtime2_5,
                y=pdf_values2_5,
                title=f'Normal Distribution for {selected_seal_version} (Total: {sum(filtered_failure_points)})',
            )
            filtered_normal.update_layout(
                xaxis_title="Days",
                yaxis_title="Normal Distribution",
                shapes=[
                    dict(
                        type='line',
                        x0=mean2_5,
                        y0=0,
                        x1=mean2_5,
                        y1=max2_5,
                        line=dict(color='red', width=2),
                    )
                ],
                annotations=[
                    dict(
                        x=mean2_5,
                        y=max2_5,
                        xref="x",
                        yref="y",
                        text="mean =" + str(round(mean2_5)) + " days",

                    )
                ], )
            filtered_survivability_fig = px.line(
                x=sorted_runtime2_5,
                y=prob_array2_5,
                title=f'Survivability Curve for {selected_seal_version} (Total: {sum(filtered_failure_points)})'
            )
            filtered_survivability_fig.update_layout(
                xaxis_title="Days",
                yaxis_title="Survival Probability",
            )
        if selected_seal_version == "ver 3.0":
            filtered_normal = px.scatter(
                x=sorted_runtime3,
                y=pdf_values3,
                title=f'Normal Distribution for {selected_seal_version} (Total: {sum(filtered_failure_points)})',
            )
            filtered_normal.update_layout(
                xaxis_title="Days",
                yaxis_title="Normal Distribution",
                shapes=[
                    dict(
                        type='line',
                        x0=mean3,
                        y0=0,
                        x1=mean3,
                        y1=max3,
                        line=dict(color='red', width=2),
                    )
                ],
                annotations=[
                    dict(
                        x=mean3,
                        y=max3,
                        xref="x",
                        yref="y",
                        text="mean =" + str(round(mean3)) + " days",

                    )
                ], )
            filtered_survivability_fig = px.line(
                x=sorted_runtime3,
                y=prob_array3,
                title=f'Survivability Curve for {selected_seal_version} (Total: {sum(filtered_failure_points)})'
            )
            filtered_survivability_fig.update_layout(
                xaxis_title="Days",
                yaxis_title="Survival Probability",
            )
        if selected_seal_version == "ver 3.0-New Screen":
            filtered_normal = px.scatter(
                x=sorted_runtime3_ns,
                y=pdf_values3_ns,
                title=f'Normal Distribution for {selected_seal_version} (Total: {sum(filtered_failure_points)})',
            ),
            filtered_normal.update_layout(
                xaxis_title="Days",
                yaxis_title="Normal Distribution",
                shapes=[
                    dict(
                        type='line',
                        x0=mean3_ns,
                        y0=0,
                        x1=mean3_ns,
                        y1=max3_ns,
                        line=dict(color='red', width=2),
                    )
                ],
                annotations=[
                    dict(
                        x=mean3_ns,
                        y=max3_ns,
                        xref="x",
                        yref="y",
                        text="mean =" + str(mean3_ns) + " days",

                    )
                ], )
            filtered_survivability_fig = px.line(
                x=sorted_runtime3_ns,
                y=prob_array3,
                title=f'Survivability Curve for {selected_seal_version} (Total: {sum(filtered_failure_points)})'
            )
            filtered_survivability_fig.update_layout(
                xaxis_title="Days",
                yaxis_title="Survival Probability",
            )
        return filtered_failure_points_fig, filtered_reason_for_pull_fig,filtered_normal,filtered_survivability_fig
    else:
        return failure_points_fig, reason_for_pull_fig, normal_dist_fig, survivability_fig

if __name__ == '__main__':
    app.run_server(debug=True)
