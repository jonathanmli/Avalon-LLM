from strategist.searchlightimprove.evolvers import BeamEvolver

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
    # include appropriate labels and title
    fig = px.box(function_df, x='category', y='score', title='Gameplay scores for each method', color='category', boxmode='overlay')
    fig.update_layout(xaxis_title='Method', yaxis_title='Score')
    # Change plotly theme to 'simple white'
    fig.update_layout(template='simple_white')
    # Hide the color legend
    fig.update_traces(showlegend=False)


    # Save the figures to the save directory
    date_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fig.write_html(os.path.join(save_dir, f'generation_vs_final_score_{date_name}.html'))

    

def main():
    save_dir = "outputs/methodbox_example"
    plot_and_save_figures(save_dir)

if __name__ == "__main__":
    main()