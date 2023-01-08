import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from tabulate import tabulate

# FOR PLOTS
import plotly.express as px
import plotly.graph_objects as go

class Funcs:

    def __init__(self, heroes: pd.Series):
        self.heroes = heroes
    
    ### FUNCTIONALITY 1 ###
    def graph_summary(self, G: nx.Graph, N: int = 10, verbose = True):
        """Prints some basic features of the graph.
        Args:
            G (nx.Graph): The graph.
            type (int): The type of graph. Can be 1 or 2.
            N (int): Denotes the top N heroes to consider.
        """
        
        if G.name not in ['First Graph', 'Second Graph']:
            ValueError('The graph name is not valid. It must be either "First Graph" or "Second Graph".')

        if verbose:
            print(f'Extracting features for the {G.name}...')
        subnodes = self.heroes.head(N).index

        # Adding comic nodes if Graph 2
        if G.name == 'Second Graph':
            comics = [node for node in G.nodes if 
                G.nodes[node]['type'] == 'comic' and                   # if the node type is comic
                len(set(nx.neighbors(G, node)) & set(subnodes)) > 0]   # and the comic contains one of the top N heroes    
            subnodes = [*subnodes, *comics]                             # Add comic nodes to subnodes 
            
        # Create subgraph
        subg = G.subgraph(subnodes)

        num_nodes = subg.number_of_nodes()
        num_edges = subg.number_of_edges()
        nwdensity = nx.density(subg)
        avedegree = np.mean(list(dict(subg.degree()).values()))
        is_sparse = nwdensity < 0.1

        q95 = np.quantile(list(dict(subg.degree()).values()), 0.99)
        hubnodes = [node for node, degree in subg.degree() if degree >= q95]

        info = {
            'Number of nodes': num_nodes,
            'Number of collaborations': num_edges,
            'Network density': nwdensity,
            'Average degree': avedegree,
            'Hub nodes': hubnodes,
            'Sparsity': is_sparse,
            'Degree distribution': dict(subg.degree())
        }

        if G.name == 'First Graph':
            info['Graph Specific Plot Data'] = dict(subg.degree(subnodes))
            info['Graph Specific Plot Name'] = 'Number of collaborations for each hero'
        else:
            info['Graph Specific Plot Data'] = dict(subg.degree(comics))
            info['Graph Specific Plot Name'] = 'Number of heroes in each comic'


        # Reporting values if verbose
        if verbose:
            print('Number of nodes:', num_nodes)
            print(f'Network density: {nwdensity:.4f}')
            print(f'Average degree: {avedegree:.4f}')
            print('10 example hub nodes:', '\t'.join(hubnodes[:10]))
            print('The network is sparse:', is_sparse)

        return info

    ### FUNCTIONALITY 2 ###
    def top_heroes(self, G: nx.Graph, node: str, metric: int, N: int = 10, verbose = True):
        """Prints the top N heroes based on the given metric.
        Args:
            G (nx.Graph): The graph.
            node (str): The node (hero or comic).
            metric (int): Integer denoting the metric. Can be 1, 2, 3 or 4, that corresponds to:
                1: Betweeness
                2: PageRank
                3: ClosenessCentrality
                4: DegreeCentrality
            N (int): Denotes the top N heroes to consider.
        """
        if metric not in [1, 2, 3, 4]:
            raise ValueError('metric must be 1, 2, 3 or 4')
        
        measure = ['Betweeness', 'PageRank', 'ClosenessCentrality', 'DegreeCentrality']
        subnodes = self.heroes[:N].index
        subg = G.subgraph(subnodes)

        if metric == 1:
            res = nx.betweenness_centrality(subg, normalized=True, weight='weight')
            print(f'{node}\'s {measure[metric-1]}: {res[node]:.4f}') if verbose else None
        elif metric == 2:
            res = nx.pagerank(G, weight='weight')
            print(f'{node}\'s {measure[metric-1]}: {res[node]:.4f}') if verbose else None
        elif metric == 3:
            res = nx.closeness_centrality(G, u=node)
            print(f'{node}\'s {measure[metric-1]}: {res:.4f}') if verbose else None
        else:
            res = nx.degree_centrality(G)
            print(f'{node}\'s {measure[metric-1]}: {res[node]:.4f}') if verbose else None

        # print(res)
        if metric == 3:
            info = {
                'Metric of node': res,
                'Name of metric': measure[metric-1],
            }
        else:
            info = {
                'Average metric': np.mean(list(res.values())),
                'Metric of node': res[node],
                'Name of metric': measure[metric-1],
            }


        return info


    ### FUNCTIONALITY 3 ###
    def shortest_route(self, G: nx.Graph, hero_seq: list, N: int = 10, verbose = True):
        """Prints the shortest route from the first hero to the last hero in the list.
        Args:
            G (nx.Graph): The graph.
            hero_seq (list): The sequence of heroes to visit.
            N (int): Denotes the top N heroes to consider.
        """
        if G.name not in ['Second Graph']:
            raise ValueError('Graph must be the second graph')

        subheroes = self.heroes[:N].index

        # Create list of comics by checking the node type and if it has a neighbor that belongs to the top N heroes
        comics = [node for node in G.nodes if G.nodes[node]['type'] == 'comic' and len(set(G.neighbors(node)).intersection(set(subheroes))) > 0]

        # Create subgraph
        subg = G.subgraph([*subheroes, *comics])

        # Check if things are going well
        print(G) if verbose else None

        # Check if the sequence of heroes are in the top N heroes
        for hero in hero_seq:
            if hero not in subg.nodes:
                print(f'A hero in the sequence is not a top {N} hero. Your lame hero is {hero}. Please try again with more popular heroes or a larger N.')
                print('Heroes must be one of the following:', *subg.nodes)
                return

        paths = []
        for i in range(len(subheroes)-1):
            paths.append(nx.shortest_path(subg, source=subheroes[i], target=subheroes[i+1]))

        print('Shortest route:', *paths)

        info = {
            'Shortest route': paths,
            'Subgraph': subg,
            'Hero sequence': hero_seq
        }

        return info

    ### FUNCTIONALITY 4 ###
    def disconnecting_graphs(self, G: nx.Graph, heroA: str, heroB: str, N: int = 10, verbose: bool = False) -> dict:
        """Prints the minimum number of links required to disconnect the original graph in two disconnected subgraphs.
        Args:
            G (nx.Graph): The graph.
            heroA (str): The first hero.
            heroB (str): The second hero.
            N (int): Denotes the top N heroes to consider.
        """

        subg = G.subgraph(self.heroes.index[:N])
        
        cutset = nx.minimum_edge_cut(subg, heroA, heroB)
        # cutset = minimum_edge_cut(subg, heroA, heroB)
        print(f'Number of links to cut to disconnect {heroA} and {heroB}: {len(cutset)}') if verbose else None

        # Check if path exists after cutting the links
        alteredg = subg.copy()
        alteredg.remove_edges_from(cutset)
        print(f'Path exists between {heroA} and {heroB}: {nx.has_path(alteredg, heroA, heroB)}') if verbose else None

        info = {'Number of links to cut': len(cutset),
                'Cutset': cutset,
                'Subgraph': subg,
                'HeroA': heroA,
                'HeroB': heroB
        }

        return info

    ### FUNCTIONALITY 5 ###
    def extracting_communities(self, G: nx.Graph, N: int = 10, hero1: str = None, hero2: str = None, verbose: bool = True, method: str = 'girvan_newman') -> dict:
        """Prints the minimum number of edges that should be removed to form communities and a list of communities.
        Args:
            G (nx.Graph): The graph.
            N (int): Denotes the top N heroes to consider.
            hero1 (str): The first hero.
            hero2 (str): The second hero.
            method (str): The method to use for extracting communities. Possible values are 'girvan_newman', 'asyn_lpa_communities', 
            'asyn_fluidc' and 'label_propagation_communities'
        """
        subnodes = self.heroes.index[:N]
        subg = G.subgraph(subnodes)

        if method == 'girvan_newman':
            res = nx.algorithms.community.centrality.girvan_newman(subg)
            communities = next(res)
        elif method == 'asyn_lpa_communities':
            res = nx.algorithms.community.asyn_lpa_communities(subg)
        elif method == 'asyn_fluidc':
            res = nx.algorithms.community.asyn_fluidc(subg, k=3)
        elif method == 'label_propagation_communities':
            res = nx.algorithms.community.label_propagation_communities(subg)
            communities = res
        else:
            print('Invalid method. Please choose one of the following: girvan_newman, asyn_lpa_communities, asyn_fluidc, label_propagation_communities')
            return

        path_exists = False
        for community in communities:
            if hero1 in community and hero2 in community:
                path_exists = True
                break

        num_edges_between_communities = 0
        edges_removed = []
        for community1, community2 in combinations(communities, 2):
            for node1 in community1:
                for node2 in community2:
                    if subg.has_edge(node1, node2):
                        num_edges_between_communities += 1
                        edges_removed.append((node1, node2))
                        edges_removed.append((node2, node1))
                

        info = {
            'Communities': communities,
            'Subgraph': subg,
            'Hero1': hero1,
            'Hero2': hero2,
            'Same community': path_exists,
            'Edges to cut': num_edges_between_communities,
            'Edges removed': edges_removed
        }

        if not verbose:
            return info


        print('Communities:') 
        for i, community in enumerate(communities):
            print(f'Community {i+1}:', *community)

        print(f'Path exists between {hero1} and {hero2}: {path_exists}')
        print(f'Number of edges to cut to form communities: {num_edges_between_communities}')
        return info

class Visualz:

    def __init__(self, heroes: pd.DataFrame):
        self.heroes = heroes
        self.funcs = Funcs(heroes)

    ### VISUALIZATION 1 ###
    def viz1(self, G, N=100):
        info = self.funcs.graph_summary(G, N=N, verbose=False)
        print(tabulate(pd.DataFrame.from_dict(info, orient='index', columns=['Value']).T[['Number of nodes', 'Number of collaborations', 'Network density', 'Average degree', 'Sparsity']], headers='keys', tablefmt='psql'))
        print('Hub nodes:', '\t'.join(info['Hub nodes'][:10]))
        fig = px.histogram(pd.DataFrame.from_dict(info['Degree distribution'], orient='index', columns=['Degree']), x='Degree', log_y=True)
        fig.update_layout(title='Degree distribution', yaxis_title='Count')
        fig.show()
        plot_data = pd.DataFrame.from_dict(info['Graph Specific Plot Data'], orient='index', columns=['Degree']).sort_values(by='Degree', ascending=False)
        plot_data = plot_data[:50] # Limiting to 50 for better visualization
        plot_data.sort_values(by='Degree', ascending=True, inplace=True)
        fig = px.bar(plot_data, x='Degree', y=plot_data.index, orientation='h')
        fig.update_layout(title=info['Graph Specific Plot Name'], yaxis_title='')
        fig.show()

    ### VISUALIZATION 2 ###
    def viz2(self, G, N, node, metric):
        info = self.funcs.top_heroes(G, node, metric, N, verbose=False)
        print(tabulate(pd.DataFrame.from_dict(info, orient='index'), headers=['Value'], tablefmt='fancy_grid'))

    ### VISUALIZATION 3 ###
    def viz3(self, G, hero_seq, N):
        info = self.funcs.shortest_route(G, hero_seq, N, verbose=False)
        comics = [t[1] for t in info['Shortest route']]
        route_df = pd.DataFrame(comics, columns=['Comic Name'], index=[f'Comic {i+1}' for i in range(len(comics))]).T
        print(tabulate(route_df, headers='keys', tablefmt='psql'))

        nodes_in_sr = list(np.unique([node for path in info['Shortest route'] for node in path]))
        edges_in_sr = [[(x[0], x[1]), (x[1], x[2]), (x[1], x[0]), (x[2], x[1])] for x in info['Shortest route']]
        # sort each tuple in the list
        edges_in_sr = [item for sublist in edges_in_sr for item in sublist]

        subg = info['Subgraph']
        subg = subg.subgraph(nodes_in_sr)

        # for edges in edges_in_sr make path = True, otherwise make path = False
        for edge in subg.edges():
            if edge in edges_in_sr:
                subg.edges[edge]['path'] = True
            else:
                subg.edges[edge]['path'] = False

        for hero in info['Hero sequence']:
            subg.nodes[hero]['seq'] = 'seq'

        # Now we implement the example for our graph subg
        pos = nx.spring_layout(subg)

        edges = {'edge_x': [],'edge_y': []}
        colored_edges = {'edge_x': [],'edge_y': []}

        for edge in subg.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            if subg.edges[edge]['path'] == True:
                colored_edges['edge_x'].append(x0)
                colored_edges['edge_x'].append(x1)
                colored_edges['edge_x'].append(None)
                colored_edges['edge_y'].append(y0)
                colored_edges['edge_y'].append(y1)
                colored_edges['edge_y'].append(None)
            else:
                edges['edge_x'].append(x0)
                edges['edge_x'].append(x1)
                edges['edge_x'].append(None)
                edges['edge_y'].append(y0)
                edges['edge_y'].append(y1)
                edges['edge_y'].append(None)
            
        edge_trace = go.Scatter(
            x=edges['edge_x'], y=edges['edge_y'],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        colored_edge_trace = go.Scatter(
            x=colored_edges['edge_x'], y=colored_edges['edge_y'],
            line=dict(width=2, color='red'),
            hoverinfo='none',
            mode='lines')

        nodes = {'node_x': [],'node_y': []}
        for node in subg.nodes():
            x, y = pos[node]
            nodes['node_x'].append(x)
            nodes['node_y'].append(y)

        node_trace = go.Scatter(
            x=nodes['node_x'], y=nodes['node_y'],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=10,
                line_width=2))

        colors = {'hero': '#F3A5AA', 'comic': '#447BBE', 'seq': '#DF1F2D'}

        node_trace.marker.color = [colors[subg.nodes[node]['seq'] if 'seq' in subg.nodes[node] else subg.nodes[node]['type']] for node in subg.nodes()]
        node_trace.text = [f'{node}: {subg.nodes[node]["type"]}' for node in subg.nodes()]

        fig = go.Figure(data=[edge_trace, colored_edge_trace, node_trace],
                        layout=go.Layout(
                            title='Shortest Route',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text=f"Sequence of heroes: {', '.join(info['Hero sequence'])}",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 ) ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        
        fig.show()

    ### VISUALIZATION 4 ###
    def viz4(self, G, N, hero1, hero2):
        info = self.funcs.disconnecting_graphs(G, hero1, hero2, N, verbose=False)
        print(f'Number of links to cut to disconnect {hero1} and {hero2}: {info["Number of links to cut"]}')
        original_graph = info['Subgraph']
        cutset = info['Cutset']
        altered_graph = original_graph.copy()
        altered_graph.remove_edges_from(cutset)

        pos = nx.spring_layout(altered_graph)
        edges = {'edge_x': [],'edge_y': []}
        for edge in altered_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edges['edge_x'].append(x0)
            edges['edge_x'].append(x1)
            edges['edge_x'].append(None)
            edges['edge_y'].append(y0)
            edges['edge_y'].append(y1)
            edges['edge_y'].append(None)
        edge_trace = go.Scatter(
            x=edges['edge_x'], y=edges['edge_y'],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        nodes = {'node_x': [],'node_y': []}
        for node in altered_graph.nodes():
            x, y = pos[node]
            nodes['node_x'].append(x)
            nodes['node_y'].append(y)
        node_trace = go.Scatter(
            x=nodes['node_x'], y=nodes['node_y'],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=10,
                line_width=2))
        # Red if heroa or herob, else blue
        node_trace.marker.color = ['red' if node in [info['HeroA'], info['HeroB']] else 'blue' for node in altered_graph.nodes()]
        # camelcase for heroname
        node_trace.text = [node.split('/')[0].title() for node in altered_graph.nodes()]
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Altered Graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40))
                            )
        fig.show()

        pos = nx.spring_layout(original_graph)
        edges = {'edge_x': [],'edge_y': []}
        for edge in original_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edges['edge_x'].append(x0)
            edges['edge_x'].append(x1)
            edges['edge_x'].append(None)
            edges['edge_y'].append(y0)
            edges['edge_y'].append(y1)
            edges['edge_y'].append(None)
        edge_trace = go.Scatter(
            x=edges['edge_x'], y=edges['edge_y'],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        nodes = {'node_x': [],'node_y': []}
        for node in original_graph.nodes():
            x, y = pos[node]
            nodes['node_x'].append(x)
            nodes['node_y'].append(y)
        node_trace = go.Scatter(
            x=nodes['node_x'], y=nodes['node_y'],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=10,
                line_width=2))
        # Red if heroa or herob, else blue
        node_trace.marker.color = ['red' if node in [info['HeroA'], info['HeroB']] else 'blue' for node in original_graph.nodes()]
        # camelcase for heroname
        node_trace.text = [node.split('/')[0].title() for node in original_graph.nodes()]
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Original Graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40))
                            )
        fig.show()
        
    ### VISUALIZATION 5 ###
    def viz5(self, G, N, heroA, heroB):
        info = self.funcs.extracting_communities(G, N, heroA, heroB, verbose=False)
        subg = info['Subgraph']
        pos = nx.spring_layout(subg)
        edges = {'edge_x': [],'edge_y': []}
        edges_removed = {'edge_x': [],'edge_y': []}
        for edge in subg.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            if edge in info['Edges removed']:
                edges_removed['edge_x'].append(x0)
                edges_removed['edge_x'].append(x1)
                edges_removed['edge_x'].append(None)
                edges_removed['edge_y'].append(y0)
                edges_removed['edge_y'].append(y1)
                edges_removed['edge_y'].append(None)
            else:
                edges['edge_x'].append(x0)
                edges['edge_x'].append(x1)
                edges['edge_x'].append(None)
                edges['edge_y'].append(y0)
                edges['edge_y'].append(y1)
                edges['edge_y'].append(None)
        edge_trace = go.Scatter(
            x=edges['edge_x'], y=edges['edge_y'],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        edge_trace_removed = go.Scatter(
            x=edges_removed['edge_x'], y=edges_removed['edge_y'],
            line=dict(width=0.5, color='red'),
            hoverinfo='none',
            mode='lines')
        nodes = {'node_x': [],'node_y': []}
        for node in subg.nodes():
            x, y = pos[node]
            nodes['node_x'].append(x)
            nodes['node_y'].append(y)
        node_trace = go.Scatter(
            x=nodes['node_x'], y=nodes['node_y'],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=10,
                line_width=2))
        colorpalette = px.colors.sequential.Plasma
        def get_color(node):
            for i, community in enumerate(info['Communities']):
                if node in community:
                    if i == 0:
                        return 'red'
                    return colorpalette[i]
        node_trace.marker.color = [get_color(node) for node in subg.nodes()]
        node_trace.text = [node.split('/')[0].title() for node in subg.nodes()]
        fig = go.Figure(data=[edge_trace, edge_trace_removed, node_trace],
                        layout=go.Layout(
                            title='Communities',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40))
                            )
        fig.show()