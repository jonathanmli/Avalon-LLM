from search_src.searchlightimprove.evolvers import BeamEvolver

import logging
import os
import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_and_save_figures(save_dir:str):
    '''
    Loads the results.csv file from the save_dir and creates the following plots:
    '''

    # load the results.csv file
    function_df = pd.read_csv(os.path.join(save_dir, 'results.csv'))

    # create a plotly express box plot where x is the category and y is the score
    # add a line for each benchmark score
    # include appropriate labels and title
    fig = px.box(function_df, x='category', y='score', title='Function Scores by Category', color='category')
    fig.update_layout(xaxis_title='Category', yaxis_title='Score')
    fig.write_html(os.path.join(save_dir, 'results.html'))

def main():
    save_dir = "outputs/good_example"
    plot_and_save_figures(save_dir)

if __name__ == "__main__":
    main()