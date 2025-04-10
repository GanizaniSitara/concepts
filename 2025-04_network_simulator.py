import networkx as nx
import random
import os
import traceback # For better error reporting

# --- Configuration for Graph Generation ---
NUM_NODES = 80  # Total number of nodes (applications)
NUM_COMMUNITIES = 6 # How many distinct communities to aim for
AVG_NODES_PER_COMMUNITY = NUM_NODES // NUM_COMMUNITIES

# Stochastic Block Model (SBM) parameters
# Sizes of the communities (must sum to NUM_NODES)
sizes = [AVG_NODES_PER_COMMUNITY] * (NUM_COMMUNITIES - 1)
sizes.append(NUM_NODES - sum(sizes)) # Add remainder to the last community

# Probability matrix (p_ij = probability of edge between community i and j)
# Higher probability along the diagonal (within community), lower off-diagonal
probs = []
p_in = 0.25   # Probability of edge WITHIN community (relatively dense)
p_out = 0.01  # Probability of edge BETWEEN communities (sparse)
for i in range(NUM_COMMUNITIES):
    row = [p_out] * NUM_COMMUNITIES # Initialize row with off-diagonal probability
    row[i] = p_in                 # Set diagonal probability
    probs.append(row)

# Output filename
output_gml_file = "input_graph.gml"

# --- Generate Graph ---
print(f"Generating graph with {NUM_NODES} nodes and {NUM_COMMUNITIES} communities...")
print(f"Community sizes: {sizes}")
print(f"Probability matrix (diagonal={p_in}, off-diagonal={p_out}):")

G = None # Initialize G
try:
    # Ensure NetworkX version compatibility for sbm
    if hasattr(nx, 'stochastic_block_model'):
        G = nx.stochastic_block_model(sizes, probs, seed=42, sparse=True)
        print("Generated graph using stochastic_block_model.")
    elif hasattr(nx, 'random_partition_graph'):
         # Fallback for potentially older NetworkX versions
         print("Warning: nx.stochastic_block_model not found directly. Trying random_partition_graph.")
         G = nx.random_partition_graph(sizes, p_in=p_in, p_out=p_out, seed=42)
         print("Generated graph using random_partition_graph.")
    else:
         print("Error: Suitable graph generation function not found in NetworkX.")
         exit() # Exit if no generation method found

    # Check if G was created successfully
    if G is None or not isinstance(G, nx.Graph):
        print("Error: Graph generation failed or did not return a valid graph object.")
        exit()

    print(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

except Exception as e:
    print(f"An error occurred during graph generation: {e}")
    traceback.print_exc() # Print full traceback for generation errors
    exit() # Exit if generation itself fails


# --- Add Node Labels (Optional but helpful) ---
print("Relabeling nodes...")
try:
    # Map integer nodes 0..N-1 to App names like AppN0, AppN1, ...
    # Iterating over existing nodes is safer than assuming range(N)
    node_mapping = {node: f"AppN{i}" for i, node in enumerate(list(G.nodes()))} # Use list() for stable iteration
    G = nx.relabel_nodes(G, node_mapping, copy=True) # Use copy=True for safety
    print("Relabeled nodes to 'AppN...' format.")
except Exception as e:
    print(f"An error occurred during node relabeling: {e}")
    # Decide whether to exit or proceed with original labels if relabeling fails
    # exit()


# --- Clean Attributes Before Saving ---
print("Cleaning node attributes before saving to GML...")
nodes_with_block_attr = 0
# Create a copy of node data items to iterate over, allows modification
node_data_items = list(G.nodes(data=True))
for node, data in node_data_items:
    # Remove the 'block' attribute added by SBM, as it's not needed downstream
    # and might contain incompatible types for GML in some cases.
    if 'block' in data:
        # Access the attribute dictionary directly via G[node] to delete
        try:
            del G.nodes[node]['block']
            nodes_with_block_attr += 1
        except KeyError:
             # Should not happen if 'block' in data is true, but handle defensively
             pass

    # Add checks for other potential non-serializable types if needed
    # e.g., if 'some_other_attr' could be a set:
    # if isinstance(data.get('some_other_attr'), set):
    #    del G.nodes[node]['some_other_attr']

print(f"Removed 'block' attribute from {nodes_with_block_attr} nodes.")

# Also check graph-level attributes (less likely source, but good practice)
graph_attrs_to_remove = []
for key, value in G.graph.items():
     if not isinstance(value, (str, int, float, bool, list, dict)): # Allow list/dict if GML supports? Check GML spec if needed. Basic types are safest.
         print(f"Warning: Removing potentially non-primitive graph attribute '{key}' of type {type(value)}")
         graph_attrs_to_remove.append(key)
for key in graph_attrs_to_remove:
     del G.graph[key]


# --- Save Graph ---
print(f"Attempting to save graph to: {output_gml_file}")
try:
    nx.write_gml(G, output_gml_file, stringizer=str) # Use stringizer for robustness
    print(f"Graph saved successfully to: {output_gml_file}")
except Exception as e:
    print(f"Error saving graph to GML: {e}")
    traceback.print_exc()