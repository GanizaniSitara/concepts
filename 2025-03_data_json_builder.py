import os
import shutil
import datetime
import math
import random
import json
import colorsys
# import pandas as pd # Not needed if input is GML
import networkx as nx
from collections import defaultdict, deque
import community as community_louvain # Requires: pip install python-louvain
import traceback # For better error reporting

# --- Configuration & Constants ---
MIN_HEXAGONS_PER_CLUSTER = 8
CLUSTER_CENTROID_GAP = 6.0
INITIAL_POS_RANGE_X = 30
INITIAL_POS_RANGE_Y = 20
SPRING_K = 0.4
SPRING_ITERATIONS = 250
TITLE_NODE_OFFSET_Q = -1
TITLE_NODE_OFFSET_R = -1
TARGET_MAX_Q = 45
TARGET_MAX_R = 35
HEX_DIRECTIONS = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]

# Seed colors for assigning to detected communities
seed_cluster_colors_list = ["#4169E1", "#DC143C", "#32CD32", "#9932CC", "#FF8C00", "#1E90FF", "#008080"]

# Input / Output Files
input_gml_file = "input_graph.gml"
output_json_file = r"c:\Solutions\Javascript\Hexmap\src\data.json" # Use raw string

# --- Helper Functions ---
def generate_harmonious_color(base_hex):
    try:
        base_hex=base_hex.lstrip("#");r,g,b=(int(base_hex[i:i+2],16)/255.0 for i in (0,2,4))
        h,l,s=colorsys.rgb_to_hls(r,g,b);h=(h+random.uniform(-0.05,0.05))%1.0
        l=max(0,min(1,l+random.uniform(-0.1,0.1)));s=max(0,min(1,s+random.uniform(-0.1,0.1)))
        r,g,b=colorsys.hls_to_rgb(h,l,s);return "#%02X%02X%02X"%(int(r*255),int(g*255),int(b*255))
    except Exception as e:
        print(f"Error generating color from {base_hex}: {e}")
        return "#CCCCCC" # Fallback grey

def euclidean_distance(pos1,pos2):
    if not isinstance(pos1,dict) or not isinstance(pos2,dict): return float('inf')
    q1,r1=pos1.get('q',0),pos1.get('r',0);q2,r2=pos2.get('q',0),pos2.get('r',0)
    return math.sqrt((q1-q2)**2+(r1-r2)**2)

def find_nearest_empty_hex(start_q,start_r,occupied_coords_set):
    start_q,start_r=int(round(start_q)),int(round(start_r))
    if (start_q,start_r) not in occupied_coords_set: return start_q,start_r
    q,r=start_q,start_r;direction_index,steps_per_side,steps_in_direction=0,1,0
    processed_in_spiral=set([(q,r)]);limit=200
    while steps_per_side<limit:
        dq,dr=HEX_DIRECTIONS[direction_index];q+=dq;r+=dr;coord=(q,r)
        if coord not in occupied_coords_set: return q,r
        if coord not in processed_in_spiral: processed_in_spiral.add(coord)
        else: print(f"Warning: Spiral search near ({start_q},{start_r}) cycle. Ret: ({q},{r})");return q,r
        steps_in_direction+=1
        if steps_in_direction==steps_per_side:
            steps_in_direction=0;direction_index=(direction_index+1)%6
            if direction_index==0:
                steps_per_side+=1;dq,dr=HEX_DIRECTIONS[direction_index];q+=dq;r+=dr;coord=(q,r)
                if coord not in occupied_coords_set: return q,r
                if coord not in processed_in_spiral: processed_in_spiral.add(coord)
                else: print(f"Warning: Spiral search near ({start_q},{start_r}) cycle on ring start. Ret: ({q},{r})");return q,r
                steps_in_direction=1
    print(f"Warning: Spiral search near ({start_q},{start_r}) limit {limit}. Ret: ({q},{r})");return q,r

# --- Load Network Graph ---
print(f"Loading graph from: {input_gml_file}")
if not os.path.exists(input_gml_file):
    print(f"Error: Input graph file not found: {input_gml_file}. Run generator first."); exit()
try:
    # GML uses labels, which should be strings now from the generator
    G_input = nx.read_gml(input_gml_file)
    print(f"Loaded graph with {G_input.number_of_nodes()} nodes and {G_input.number_of_edges()} edges.")
except Exception as e:
    print(f"Error loading graph: {e}"); exit()

# --- Community Detection ---
print("Performing community detection using Louvain method...")
try:
    partition = community_louvain.best_partition(G_input, random_state=42)
    num_communities = len(set(partition.values()))
    if num_communities == 0:
         print("Warning: No communities detected. Treating all nodes as one community.")
         # Assign all nodes to community 0 if none detected
         partition = {node: 0 for node in G_input.nodes()}
         num_communities = 1
    print(f"Detected {num_communities} communities.")
    community_counts = defaultdict(int)
    for node,comm_id in partition.items(): community_counts[comm_id]+=1
    print("Community sizes:", dict(sorted(community_counts.items())))
except ImportError: print("Error: 'community' (python-louvain) library not found. Install: pip install python-louvain"); exit()
except Exception as e: print(f"Error during community detection: {e}"); exit()

# --- Build Initial Data Structures ---
print("Building initial data structures from graph and communities...")
app_nodes = {}
clusters = defaultdict(lambda: {"id":"", "name":"", "color":"", "hexCount":0, "gridPosition":{"q":0,"r":0},
                                "department":"Discovered", "leadName":"N/A", "leadEmail":"n/a",
                                "budgetStatus":"N/A", "priority":"N/A", "lastUpdated":datetime.date.today().isoformat(),
                                "description":"Community discovered via Louvain.", "applications":[]})
community_colors = {}
active_group_ids = set() # Store string community IDs like "Comm0"

# Assign community IDs and colors
for node_id_str in G_input.nodes(): # Ensure using string IDs from graph
    community_id_raw = partition.get(node_id_str, -1)
    community_id = f"Comm{community_id_raw}" # Use string format "CommX"
    active_group_ids.add(community_id)

    if community_id not in community_colors:
         idx = len(community_colors)
         if idx < len(seed_cluster_colors_list): base_color = seed_cluster_colors_list[idx]
         else: base_color = random.choice(seed_cluster_colors_list)
         community_colors[community_id] = generate_harmonious_color(base_color)

    app_color = community_colors[community_id]
    connections = [{"to": str(neighbor), "type": "link", "strength": "medium"}
                   for neighbor in G_input.neighbors(node_id_str)]

    app_nodes[node_id_str] = {
        "id": node_id_str, "name": node_id_str, "description": f"Node in {community_id}",
        "color": app_color, "gridPosition": None, "connections": connections,
        "_temp_cluster_id": community_id
    }

# Populate cluster metadata
for gid in active_group_ids:
    if gid not in clusters:
         clusters[gid]["id"]=gid; clusters[gid]["name"]=gid
         clusters[gid]["description"]=f"Community {gid} discovered via Louvain."
         clusters[gid]["color"]=community_colors.get(gid,"#CCCCCC")

# --- Global Force-Directed Layout ---
print(f"Generating initial random positions...")
initial_pos = {nid:(random.uniform(-INITIAL_POS_RANGE_X,INITIAL_POS_RANGE_X),
                    random.uniform(-INITIAL_POS_RANGE_Y,INITIAL_POS_RANGE_Y)) for nid in G_input.nodes()}
print(f"Performing global spring layout (k={SPRING_K}, iterations={SPRING_ITERATIONS})...")
raw_positions = nx.spring_layout(G_input, pos=initial_pos, k=SPRING_K, iterations=SPRING_ITERATIONS, seed=42)

# Store initial float positions (no scaling yet)
print("Storing initial float positions...")
for app_id,pos in raw_positions.items():
     if app_id in app_nodes: app_nodes[app_id]["gridPosition"]={"q":pos[0],"r":pos[1]}

# --- Adjust Cluster Positions ---
print("Adjusting cluster positions...")
def calculate_centroid(cluster_id,current_app_nodes):
    nodes=[n for n in current_app_nodes.values() if n.get("_temp_cluster_id")==cluster_id and n.get("gridPosition")]
    if not nodes: return None
    q_sum=sum(n["gridPosition"].get("q",0) for n in nodes);r_sum=sum(n["gridPosition"].get("r",0) for n in nodes)
    c=len(nodes);return {"q":q_sum/c,"r":r_sum/c} if c>0 else None
adjusted=True;iteration=0;max_iterations=250
while adjusted and iteration<max_iterations:
    adjusted=False;current_centroids={gid:calculate_centroid(gid,app_nodes) for gid in active_group_ids}
    valid_centroids={gid:pos for gid,pos in current_centroids.items() if pos};sorted_gids=sorted(list(valid_centroids.keys()))
    for i in range(len(sorted_gids)):
        for j in range(i+1,len(sorted_gids)):
            gid1,gid2=sorted_gids[i],sorted_gids[j];pos1,pos2=valid_centroids[gid1],valid_centroids[gid2]
            dist=euclidean_distance(pos1,pos2)
            if 0<dist<CLUSTER_CENTROID_GAP:
                overlap=CLUSTER_CENTROID_GAP-dist;push_force=overlap/2.0;dx,dy=pos2.get('q',0)-pos1.get('q',0),pos2.get('r',0)-pos1.get('r',0)
                norm=math.sqrt(dx*dx+dy*dy);
                if norm==0: norm=1;dx,dy=random.uniform(-.5,.5),random.uniform(-.5,.5)
                push_q,push_r=(dx/norm)*push_force,(dy/norm)*push_force
                for data in app_nodes.values():
                    if data.get("gridPosition"):
                        if data.get("_temp_cluster_id")==gid1: data["gridPosition"]["q"]-=push_q;data["gridPosition"]["r"]-=push_r
                        elif data.get("_temp_cluster_id")==gid2: data["gridPosition"]["q"]+=push_q;data["gridPosition"]["r"]+=push_r
                adjusted=True
            elif dist==0:
                 q1,r1=random.uniform(-.5,.5)*CLUSTER_CENTROID_GAP/2,random.uniform(-.5,.5)*CLUSTER_CENTROID_GAP/2
                 q2,r2=random.uniform(-.5,.5)*CLUSTER_CENTROID_GAP/2,random.uniform(-.5,.5)*CLUSTER_CENTROID_GAP/2
                 for data in app_nodes.values():
                     if data.get("gridPosition"):
                         if data.get("_temp_cluster_id")==gid1: data["gridPosition"]["q"]+=q1;data["gridPosition"]["r"]+=r1
                         elif data.get("_temp_cluster_id")==gid2: data["gridPosition"]["q"]+=q2;data["gridPosition"]["r"]+=r2
                 adjusted=True
    iteration+=1
if iteration==max_iterations: print(f"Warning: Centroid adjustment reached max iterations.")
else: print(f"Centroid adjustment completed in {iteration} iterations.")

# --- Round coords & Resolve Overlaps ---
print("Rounding coordinates and resolving overlaps...")
# Initial rounding pass (stores integer coords)
for node_data in app_nodes.values():
    if node_data.get("gridPosition"):
        node_data["gridPosition"]["q"]=int(round(node_data["gridPosition"]["q"]))
        node_data["gridPosition"]["r"]=int(round(node_data["gridPosition"]["r"]))
# Resolution passes
resolved_count=0;needs_resolve=True;resolve_iter=0;max_resolve_iter=10
while needs_resolve and resolve_iter<max_resolve_iter:
    needs_resolve=False;iter_resolved=0;node_list=list(app_nodes.values());random.shuffle(node_list)
    current_map={(n["gridPosition"]["q"],n["gridPosition"]["r"]):n["id"] for n in node_list if n.get("gridPosition")}
    occupied=set(current_map.keys())
    for n_data in node_list:
        if not n_data.get("gridPosition"):continue
        q,r=n_data["gridPosition"]["q"],n_data["gridPosition"]["r"];coord=(q,r)
        if coord in occupied and current_map.get(coord)!=n_data["id"]:
             nq,nr=find_nearest_empty_hex(q,r,occupied)
             if (nq,nr)!=coord:
                 n_data["gridPosition"]["q"]=nq;n_data["gridPosition"]["r"]=nr
                 if current_map.get(coord)==n_data["id"]:del current_map[coord]
                 occupied.discard(coord);occupied.add((nq,nr));current_map[(nq,nr)]=n_data["id"]
                 resolved_count+=1;iter_resolved+=1;needs_resolve=True
    resolve_iter+=1;print(f"Overlap resolution pass {resolve_iter}, resolved: {iter_resolved}")
all_occupied_coords=set((n["gridPosition"]["q"],n["gridPosition"]["r"]) for n in app_nodes.values() if n.get("gridPosition"))
print(f"Total overlaps resolved after rounding: {resolved_count}")

# --- Place Dummy Nodes ---
print("Placing dummy nodes...")
dummy_nodes_added={}
for gid in active_group_ids:
    if gid not in clusters:continue
    c_nodes=[n for n in app_nodes.values() if n.get("_temp_cluster_id")==gid and n.get("gridPosition")]
    n_real=len(c_nodes);n_dummy=max(0,MIN_HEXAGONS_PER_CLUSTER-n_real)
    if n_dummy>0:
        f_centroid=calculate_centroid(gid,app_nodes);
        if not f_centroid:continue
        for i in range(n_dummy):
            d_id=f"{gid}_dummy_{i}";dq,dr=find_nearest_empty_hex(f_centroid['q'],f_centroid['r'],all_occupied_coords)
            d_node={"id":d_id,"name":"","description":"Placeholder","color":clusters[gid]["color"],
                    "gridPosition":{"q":dq,"r":dr},"connections":[],"_temp_cluster_id":gid,"_is_dummy":True}
            dummy_nodes_added[d_id]=d_node;all_occupied_coords.add((dq,dr))

# --- Place Title Nodes ---
print("Placing title nodes relative to content bounds...")
title_nodes_added={}
all_placed_nodes={**app_nodes,**dummy_nodes_added}
sorted_gids=sorted(active_group_ids,key=lambda g:clusters[g].get("name",""))
for gid in sorted_gids:
    if gid not in clusters:continue
    c_nodes=[n for n in all_placed_nodes.values() if n.get("_temp_cluster_id")==gid and n.get("gridPosition") and not n.get("isTitle")]
    minq,maxq,minr,maxr=float('inf'),float('-inf'),float('inf'),float('-inf')
    if c_nodes:
        for n in c_nodes:pos=n["gridPosition"];minq=min(minq,pos.get('q',0));maxq=max(maxq,pos.get('q',0));minr=min(minr,pos.get('r',0));maxr=max(maxr,pos.get('r',0))
    else: cent=calculate_centroid(gid,all_placed_nodes);minq=cent['q'] if cent else 0;minr=cent['r'] if cent else 0
    tq=minq+TITLE_NODE_OFFSET_Q;tr=minr+TITLE_NODE_OFFSET_R
    fq,fr=find_nearest_empty_hex(tq,tr,all_occupied_coords)
    t_id=f"{gid}_title";
    t_node={"id":t_id,"name":clusters[gid]["name"],"description":f"Title node for {clusters[gid]['name']}",
            "color":clusters[gid]["color"],"gridPosition":{"q":fq,"r":fr},
            "connections":[],"isTitle":True,"_temp_cluster_id":gid}
    title_nodes_added[t_id]=t_node;all_occupied_coords.add((fq,fr))
    clusters[gid]["gridPosition"]=t_node["gridPosition"]

# --- Final Scaling and Centering ---
print("Applying final scaling and centering...")
final_nodes={**app_nodes,**dummy_nodes_added,**title_nodes_added}
minq,maxq,minr,maxr=float('inf'),float('-inf'),float('inf'),float('-inf');n_bounds=0
for n_data in final_nodes.values():
    if n_data.get("gridPosition"):
        pos=n_data["gridPosition"];q,r=pos.get('q',0),pos.get('r',0)
        minq=min(minq,q);maxq=max(maxq,q);minr=min(minr,r);maxr=max(maxr,r);n_bounds+=1
if n_bounds>0 and maxq>minq and maxr>minr:
    w=maxq-minq;h=maxr-minr;cq=(minq+maxq)/2.0;cr=(minr+maxr)/2.0
    sq=(TARGET_MAX_Q*2)/w if w>0 else 1;sr=(TARGET_MAX_R*2)/h if h>0 else 1
    f_scale=min(sq,sr,1.5);print(f"Final Scale Applied: {f_scale:.3f}")
    for n_data in final_nodes.values():
        if n_data.get("gridPosition"):
            pos=n_data["gridPosition"];oq,oldr=pos.get('q',0),pos.get('r',0)
            scq=(oq-cq)*f_scale;scr=(oldr-cr)*f_scale
            pos["q"]=int(round(scq));pos["r"]=int(round(scr))
else: print("Warning: Could not determine bounds for final scaling.")

# --- Final Assembly ---
print("Assembling final JSON data...")
final_data=[]
for gid,c_data in sorted(clusters.items(),key=lambda i:i[1].get("name","")):
    apps=[];t_node=title_nodes_added.get(f"{gid}_title")
    if t_node: apps.append(t_node)
    for n_id,n_data in final_nodes.items():
        if n_data.get("_temp_cluster_id")==gid and not n_data.get("isTitle"):
            if n_data.get("gridPosition"): apps.append(n_data)
            else: print(f"Warning: Node {n_id} missing gridPosition.")
    c_data["applications"]=apps;c_data["hexCount"]=len(apps)-(1 if t_node else 0)
    if t_node and t_node.get("gridPosition"):c_data["gridPosition"]=t_node["gridPosition"]
    final_data.append(c_data)

# --- Final Cleanup ---
print("Cleaning up temporary node data...")
for c in final_data:
    for n in c.get("applications",[]):
        if isinstance(n,dict): n.pop("_temp_cluster_id",None);n.pop("_is_dummy",None)

# --- Write Output ---
print("Finalizing JSON output...")
json_output={"clusters":final_data}
# Debug print... (optional)
backup_dir=os.path.dirname(output_json_file);os.makedirs(backup_dir,exist_ok=True)
if os.path.exists(output_json_file):
    ts=datetime.datetime.now().isoformat(timespec="minutes").replace(":","-");bf=f"data_{ts}.json";bp=os.path.join(backup_dir,bf)
    try: shutil.copy(output_json_file,bp);print(f"Backed up data to: {bp}")
    except Exception as e: print(f"Error backing up file: {e}")
print(f"Writing final JSON data to: {output_json_file}")
try:
    with open(output_json_file,"w") as f: json.dump(json_output,f,indent=2);print("JSON data written successfully.")
except Exception as e: print(f"Error writing JSON file: {e}")

print("Processing complete.")
(input_gml_file, output_json_file)