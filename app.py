import requests
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
dict1={
    "drought":"https://eonet.gsfc.nasa.gov/api/v3/categories/drought?days=100",
    "dustHaze":"https://eonet.gsfc.nasa.gov/api/v3/categories/dustHaze?days=100",
    "earthquakes":"https://eonet.gsfc.nasa.gov/api/v3/categories/earthquakes?days=100",
    "floods":"https://eonet.gsfc.nasa.gov/api/v3/categories/floods?days=100",
    "landslides":"https://eonet.gsfc.nasa.gov/api/v3/categories/landslides?days=100",
    "manMade":"https://eonet.gsfc.nasa.gov/api/v3/categories/manmade?days=100",
    "seaLakeice":"https://eonet.gsfc.nasa.gov/api/v3/categories/seaLakeIce?days=100",
    "severeStorms":"https://eonet.gsfc.nasa.gov/api/v3/categories/severeStorms?days=100",
    "snow":"https://eonet.gsfc.nasa.gov/api/v3/categories/snow?days=100",
    "tempExtremes":"https://eonet.gsfc.nasa.gov/api/v3/categories/tempExtremes?days=100",
    "volcanoes":"https://eonet.gsfc.nasa.gov/api/v3/categories/volcanoes?days=100",
    "waterColor":"https://eonet.gsfc.nasa.gov/api/v3/categories/waterColor?days=100",
    "wildFires":"https://eonet.gsfc.nasa.gov/api/v3/categories/wildfires?days=100"
}
dict2,dict3 = {},{}


for i in dict1:
    r = requests.get(dict1[i])
    dict2[i] = len(r.json()['events'])


for i in dict2:
    if dict2[i] != 0:
        dict3[i] = dict2[i]

""""""


app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Categories of incidents": list(dict3.keys()),
    "Incidents (100 days)": list(dict3.values())
})

fig = px.bar(df, x="Categories of incidents", y="Incidents (100 days)", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)