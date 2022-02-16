import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import datetime
import json
import pandas as pd
import plotly
import io
import numpy as np
import base64
from base64 import decodestring
import os
import glob
import shutil
from jupyter_pipeline import *

# Initialise the app
app = dash.Dash(__name__)

def upload():
    obj = dcc.Upload(id='upload-image',children=html.Div([html.A('Upload Image',
               style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        html.H3(id = 'label',style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },multiple=True)
    return obj

def message():
    obj = html.Div(id='output-image-upload',
        style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
    return obj

def image_option():
    obj = html.Div([dcc.RadioItems(id = 'user_img_option',
                   options = [{'label': 'Original Image' ,'value':'original'},
                              {'label': 'Compose Box' ,'value':'compose'},
                              {'label': 'Symbol Box' ,'value':'symbol'}],labelStyle={'display': 'block'})])
    return obj

def loading_gif():
    obj = dcc.Loading(id="loading-1", children=[
        html.Div(id='output-image-upload',
            style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center','margin-top':"150px"})], type="cube")
    return obj

def div(width):
    obj = html.Div(style={'width': width, 'display': 'inline-block'},children = [html.H3()])
    return obj 


def left():
    children = [html.H1('Mail Classifer'),upload(),html.Br(),html.H2('Select option for preview'),image_option()]
    return children

def newline():
    return html.Br()

def right():
    # children = [newline(),newline(),newline(),newline(),newline(),newline(),newline(),newline(),newline(),
    #             loading_gif(),html.Div(id='output-image-1')]
    children = [html.Br(),html.Br(),loading_gif(),html.Div(id='output-image-1')]
    return children

# Define the app
app.layout = html.Div(children=[
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(className='four columns div-user-controls',children = left()), 
                                  html.Div(id = 'right' ,className='eight columns div-for-charts bg-grey',children = right()) 
                                  ])
                                ])

def parse(fname,width,height):
    try:
        test_base64 = base64.b64encode(open(fname, 'rb').read()).decode('ascii')
    except:
        return
    return html.Div([html.Img(src='data:image/png;base64,{}'.format(test_base64),style={
            'width': width,
            'height': height,            
        })], style={'textAlign': 'center'})

@app.callback(Output('output-image-upload', 'children'), [Input('upload-image', 'contents')])
def update_output(images):
    if not images:
        return ''
    try:
        shutil.rmtree('temp')
    except:
        pass
    for i, image_str in enumerate(images):
        image = image_str.split(',')[1]
        data = decodestring(image.encode('ascii'))
        
        if not os.path.isdir('temp'):
            os.mkdir('temp')
        with open("temp/image{}.jpg".format(i), "wb") as f:
            f.write(data)
        text = detect("temp/image{}.jpg".format(i))
        with open("temp/original{}.jpg".format(i), "wb") as f:
            f.write(data)
    return [html.H1('Image is classified as {}'.format(text))]

@app.callback(Output('output-image-1', 'children'), [Input('user_img_option', 'value'),Input('output-image-upload', 'children')])
def update_output(opt,contents):
    if not opt and not contents:
        return
    original_file,symbol_file,compose_file = '','',''
    for i in glob.glob('temp/*'):
        if 'original' in i:
            original_file = i
        elif '_' in i:
            symbol_file = i
        else:
            compose_file = i

    if opt == 'original':
        children = [parse(original_file,width = '80%',height = '60%')]
        return children
    elif opt == 'compose':
        children = [parse(compose_file,width = '40%',height = '30%')]
        return children
    elif opt == 'symbol':
        children = [parse(symbol_file,width = '10%',height = '10%')]
        return children

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,dev_tools_ui=True)