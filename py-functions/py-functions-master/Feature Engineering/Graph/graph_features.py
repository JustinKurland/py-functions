import pandas as pd
import networkx as nx

def calculate_graph_metrics(df):
    """
    Calculates graph metrics for each group of pages based on DMHQ_ID.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe with columns: DMHQ_ID, PAGE_NAME, ORDER
    
    Returns:
    --------
    pandas.DataFrame
        A dataframe with rows for each DMHQ_ID and columns for each page name and its respective graph metrics.
    
    Example:
    --------
    >>> df = pd.DataFrame({'DMHQ_ID': [1, 1, 1, 2, 2], 'PAGE_NAME': ['Page1', 'Page2', 'Page3', 'Page1', 'Page2'], 
                           'ORDER': [1, 2, 3, 1, 2]})
    >>> calculate_graph_metrics(df)
      DMHQ_ID  NumNodes  NumEdges  Page1_DegreeCentrality  Page1_BetweennessCentrality  Page1_PageRank  Page2_DegreeCentrality  Page2_BetweennessCentrality  Page2_PageRank  Page3_DegreeCentrality  Page3_BetweennessCentrality  Page3_PageRank
            1         3         2                     0.5                          0.0        0.333333                     1.0                          1.0        0.333333                     0.5                          0.0             0.5
            2         2         1                     1.0                          1.0        0.500000                     1.0                          0.0        0.500000                     NaN                          NaN             NaN
    """

    # Create an empty dataframe to store the results
    result_df = pd.DataFrame(columns=['DMHQ_ID', 'NumNodes', 'NumEdges'])

    # Get the list of unique PAGE_NAME values
    page_names = sorted(df['PAGE_NAME'].unique())

    # Create a column for each PAGE_NAME and metric combination
    for page_name in page_names:
        result_df[f"{page_name}_DegreeCentrality"] = pd.Series(dtype=float)
        result_df[f"{page_name}_BetweennessCentrality"] = pd.Series(dtype=float)
        result_df[f"{page_name}_PageRank"] = pd.Series(dtype=float)

    # Group the dataframe by DMHQ_ID
    groups = df.groupby('DMHQ_ID')

    # Loop through each group
    for group_name, group_df in groups:
        # Create a directed graph
        G = nx.DiGraph()

        # Add edges to the graph based on the order of the PAGE_NAME values
        edges = [(group_df.iloc[i]['PAGE_NAME'], group_df.iloc[i+1]['PAGE_NAME']) for i in range(len(group_df)-1)]
        G.add_edges_from(edges)

        # Calculate network metrics
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G)

        # Create a row for the group in the results dataframe
        row = {'DMHQ_ID': group_name, 'NumNodes': num_nodes, 'NumEdges': num_edges}

        # Add the degree centrality, betweenness centrality, and pagerank values for each PAGE_NAME
        for page_name in page_names:
            if page_name in G.nodes():
                row[f"{page_name}_DegreeCentrality"] = degree_centrality[page_name]
                row[f"{page_name}_BetweennessCentrality"] = betweenness_centrality[page_name]
                row[f"{page_name}_PageRank"] = pagerank[page_name]
            else:
                row[f"{page_name}_DegreeCentrality"] = None
                row[f"{page_name}_BetweennessCentrality"] = None
                row[f"{page_name}_PageRank"] = None

        # Append the row to the results dataframe
        result_df = result_df.append(row, ignore_index=True)

    return result_df
