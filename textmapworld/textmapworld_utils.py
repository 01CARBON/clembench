import ast
import os
import random
import networkx as nx
import numpy as np
from graph_generator import GraphGenerator
from mapworld.ade_maps import ADEMap

"----------------------------------------------------"
"The functions used in instance_generator.py"

def generate_filename(game_type, graph_size, cycle_type, ambiguity):

    if cycle_type == "cycle_true":
        cycle = "True"
    elif cycle_type == "cycle_false":
        cycle = "False"
    if graph_size==None:
        filename_parts = [game_type.capitalize().split("_graph")[0], cycle, str(ambiguity)]
        filename = "_".join(filename_parts)
    else:
        filename_parts = [game_type.capitalize().split("_graph")[0], str(graph_size), cycle, str(ambiguity)]
        filename = "_".join(filename_parts) + ".txt"
    return filename


def load_check_graph(file_graphs, instance_number, game_type):
    grids = []
    with open(str(file_graphs), 'r') as file:
        for c, line in enumerate(file):
            line = line.rstrip()
            doc = ast.literal_eval(line)
            nodes = doc.get('Graph_Nodes', [])
            check_set = set()
            if c < instance_number:
                if all(isinstance(item, tuple) for item in nodes):
                    checked_graph_type = "unnamed_graph"
                    check_set.add(checked_graph_type)
                elif all(isinstance(item, str) for item in nodes):
                    checked_graph_type = "named_graph"
                    check_set.add(checked_graph_type)
                grids.append(doc)
        if  check_set != {str(game_type)}:
            raise ValueError("Graph type does not match the specified type")
    return grids


def generate_result_file(graph_metadata: dict, G, **kwargs) -> dict:
    """
    Create textmapworld compatible results file instance (ref textmapworld/textmapworld_main/files/*.txt)

    Args:
        graph_metadata: A dictionary representing the graph definitions liek edges, nodes, n_rooms, .

    Return:
        result: A dict containing graph metadata
    """
    graph_dirs = graph_metadata["directions"]
    dir_list = []
    for k in graph_dirs.keys():
        dir_list.append((k, graph_dirs[k]))

    graph_moves = graph_metadata["moves"]
    moves_list = []
    for k in graph_moves.keys():
        moves_list.append(
            {
                "node": k,
                "node_moves": graph_moves[k]
            }
        )

    results_dict = {
        "Picture_Name": "graph_" + graph_metadata['graph_id'] + ".png",
        "Graph_Type": kwargs.get("graph_type", None),
        "Grid_Dimension": graph_metadata['m'], # m=n for textmapworld
        "Graph_Nodes": graph_metadata["graph_nodes"],
        "Graph_Edges": graph_metadata["graph_edges"],
        "N_edges": len(graph_metadata["graph_edges"]),
        "Initial_Position": np.random.choice(graph_metadata["graph_nodes"]),
        "Directions": dir_list,
        "Moves": moves_list,
        "Cycle": kwargs.get("cycle_bool", None),
        "Ambiguity": kwargs.get("ambiguity", None),
        "Mapping": graph_metadata['mapping']
    }

    print(f"Graph_type - {results_dict['Graph_Type']}")

    return results_dict



def create_graphs_file(graphs_file_name, num_graphs, graph_type, n, m, rooms, cycle_bool, abiguity, game_name):
    
    generated_graphs = 0
    with open(graphs_file_name, "w") as f:
        file_length = os.path.getsize(graphs_file_name)
        while generated_graphs < num_graphs:
            # new_instance = GraphGenerator(graph_type, n, m, rooms, cycle_bool, abiguity, game_name)
            # result = new_instance.generate_instance()

            ade_map = ADEMap(m, n, rooms)
            if cycle_bool:
                G = ade_map.create_cyclic_graph(n_loops=1)
            else:
                G = ade_map.create_acyclic_graph()

            G = ade_map.assign_types(G, ambiguity=[1])
            G = ade_map.assign_images(G)

            graph_metadata = ade_map.metadata(G)
            result = generate_result_file(graph_metadata, G, cycle_bool=cycle_bool, ambiguity=abiguity, graph_type=graph_type)

            if result:
                f.write(str(result) + "\n")
                generated_graphs+=1
    # Check if file is empty
    if os.path.getsize(graphs_file_name) == 0:
        raise ValueError("Generated file is empty")
    return graphs_file_name

"----------------------------------------------------"
"The functions used in master.py"

def get_directions(node, direction_list, saved_node):

    if saved_node != node:
        node = saved_node
    node_directions = None  
    for i in direction_list:
        if i[0]==node:
            node_directions=i[1]
            break
    return node_directions

def get_directions_main(node, direction_list, saved_node, graph_type):

    print(f"Node: {node}, \nDirection list: {direction_list}, \nsaved_node: {saved_node}, \ngraph_type: {graph_type}")

    if saved_node != node:
        node = saved_node
    node_directions = None  
    for i in direction_list:
        if graph_type == "named_graph":
            if i[0].lower()==node.lower():
                node_directions=i[1]
                break
        elif graph_type == "unnamed_graph":
            if i[0]==node:
                node_directions=i[1]
                break
    print(f"\n Node Directions : {node_directions}")
    return node_directions

def string_available_directions(word_list):
    return ', '.join(word_list)

def have_common_element(str1, str2):
    common_elements = ["east", "west", "north", "south"]
    # Convert strings to sets of words
    words1 = set(str1.split())
    words2 = set(str2.split())
    # Check for exact matches
    common_matches = words1.intersection(words2).intersection(common_elements)
    # Return True if there is at least one exact match
    return any(match in common_elements for match in common_matches)


def get_nextnode_label(moves, node, utterance, move_construction):
    next_label=None
    utterance = utterance.strip()
    for move in moves:
        if move["node"]==node:
            moves_node=move['node_moves']
            for step in moves_node:
                move_type = step[0]
                if move_type ==utterance:
                    next_label=step[1]
    return next_label, move_type

def ambiguity_move(old_one, new_one, mapping, moves, move_type):
    
    mapping_dict, label_moves = mapping
    for label, name in mapping_dict.items():
        if name == old_one:
            for move in moves:
                if name == move["node"]:
                    for label_move in label_moves:
                        if label_move["node"] == label:
                            new_labelled_moves = [(m[0], mapping_dict[m[1]]) for m in label_move["node_moves"]]
                            for m in new_labelled_moves:
                                if m[1] == new_one and m[0] == move_type:
                                    for s in label_move["node_moves"]:
                                        if s[0] == move_type:
                                            return label, s[1]
                                        
def loop_identification(visited_nodes):
    if len(visited_nodes) >= 4:
        if len(set(visited_nodes[-4:])) < 3:
            return True
    return False

"----------------------------------------------------"
"The functions used in instance_generator.py"

# Function to get nodes at a certain distance from the initial node
def get_nodes_at_distance(graph, initial_node, distance):
    return [node for node in nx.single_source_shortest_path_length(graph, initial_node) if nx.single_source_shortest_path_length(graph, initial_node)[node] == distance]

# Randomly select nodes at various distances from the initial node
def select_nodes_at_distances(G, initial_position, max_distance):
    chosen_nodes = {}
    for distance in range(max_distance):
        nodes_at_distance = get_nodes_at_distance(G, initial_position, distance)
        if nodes_at_distance:
            random_node = random.choice(nodes_at_distance)
            chosen_nodes[str(distance)] = random_node
        else:
            print(f"No nodes found at distance {distance} from {initial_position}")
    return chosen_nodes

def lowercase_list_strings(original_list):
    return [item.lower() for item in original_list]

def lowercase_tuple_strings(original_list, type):
    if type == "generated":
        combined_list = [value for sublist in original_list.values() for value in sublist]
        return [(item[0].lower(), item[1].lower()) for item in combined_list]
    elif type == "original":
        return [(item[0].lower(), item[1].lower()) for item in original_list]
    elif type == "none":
        return original_list
    
def create_graph(nodes, edges, type):
    G = nx.Graph()
    if type == "generated" or type == "original":
        nodes= lowercase_list_strings(nodes)
    edges= lowercase_tuple_strings(edges, type)
    G.add_nodes_from(nodes)
    for edge in edges:
        if len(edge) == 2:
            G.add_edge(edge[0], edge[1])
    return G

#Create a networkx graph from the given graph data.
def create_graph_specificroom(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G
    
def normalize(distance):
        normalized_distance = 1 / (1 + np.exp(-0.5 * distance))
        normalized_distance = 2*(normalized_distance - 0.5)
        return normalized_distance
    
def calculate_similarity(graph1, graph2):
    distance = nx.graph_edit_distance(graph1, graph2)
    normalized_distance = normalize(distance)
    similarity = 1 - normalized_distance
    return similarity

def count_word_in_sentence(sentence, word):
    # Split the sentence into words
    words = sentence.split()
    
    # Convert the words and the target word to lowercase for case insensitive comparison
    word = word.lower()
    words = [w.lower() for w in words]
    
    # Count the occurrences of the word
    count = words.count(word)
    return count