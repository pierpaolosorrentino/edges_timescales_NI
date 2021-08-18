import numpy as np
import matplotlib.pyplot as plt  
import scipy.io
import networkx as nx
import scipy.io
#import upsidedown

def listToDict(lst):
    op = { i : lst[i] for i in range(0, len(lst) ) }
    return op

def drawgraph(matr,w):
    G=nx.from_numpy_matrix(matr)#,create_using=nx.DiGraph)
    lab=listToDict(SC_labelsR+SC_labelsL[::-1])
    xrange=range
    node_list = sorted(G.nodes())
    angle = []
    angle_dict = {}  
    for i, node in zip(xrange(nregions),node_list):
        theta = np.pi/2-2.0*np.pi*i/nregions-np.pi/nregions
        angle.append((np.cos(theta),np.sin(theta)))
        angle_dict[node] = theta           
    pos = {}
    for node_i, node in enumerate(node_list[:int(len(node_list)/2)]):
        pos[node] = angle[node_i]
        
    for node_i, node in enumerate(node_list[int(len(node_list)/2):]):
        pos[node] = angle[-node_i-1]
    
    # figsize is intentionally set small to condense the graph
    fig, ax = plt.subplots(figsize=(28,28))
    margin=0.33
    fig.subplots_adjust(margin, margin, 1.-margin, 1.-margin)
    ax.axis('equal')

    nx.draw(G, width=w, pos=pos, with_labels=False, edge_color='#909090', ax=ax, node_color=nodecolor, node_shape='o',node_size=150)
    #description = nx.draw_networkx_labels(G, pos, labels=lab, font_color='k')

#     r = fig.canvas.get_renderer()
#     trans = plt.gca().transData.inverted()
#     for node, t in description.items():
#         bb = t.get_window_extent(renderer=r)
#         bbdata = bb.transformed(trans)
#         radius = 1.2+bbdata.width/2.
#         position = (radius*np.cos(angle_dict[node]),radius* np.sin(angle_dict[node]))
#         t.set_position(position)
#         t.set_rotation(angle_dict[node]*360.0/(2.0*np.pi))
#         t.set_clip_on(False)
        
    ax= plt.gca()
    ax.collections[0].set_edgecolor(fon)
    plt.savefig('AALnet.png', transparent=True, format="PNG", dpi=140)
    plt.show()
    
    
with open('/Users/giovanni/Desktop/Neuro/Pierpaolo-MEG/labels.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
SC_lab = [x.strip() for x in content] 
SC_lab=SC_lab[:78]
nregions = len(SC_lab)     #number of regions

# SC_labelsR=SC_lab[:int(nregions/2)]
# SC_labelsL=SC_lab[int(nregions/2):]
# for i in range(7):
#     sci=SC_labelsR[21];SC_labelsR.remove(sci);SC_labelsR.append(sci)
#     sci=SC_labelsL[21];SC_labelsL.remove(sci);SC_labelsL.append(sci)
# SC_labels=SC_labelsR+SC_labelsL

# Neword_lab=np.zeros(len(SC_lab),dtype=np.int)
# for i in range(nregions):
#     Neword_lab[i]=np.where(np.asarray(SC_lab)==SC_labels[i])[0]
    
Ordine2021L=np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20,28, 29, 30, 31, 32, 33, 34,27, 35, 36, 37, 38, 21, 22,
       23, 24, 25, 26])
Ordine2021=np.append(Ordine2021L,Ordine2021L+39)

SC_labelsR=list(np.asarray(SC_lab)[Ordine2021L+39])

SC_labelsL=list(np.asarray(SC_lab)[Ordine2021L])

SC_labels=SC_labelsR+SC_labelsL


# Import structural connectivity and define it via the log
atl = scipy.io.loadmat('/Users/giovanni/Desktop/Neuro/Pierpaolo-MEG/dti_naples_1009.mat')
SC=atl['D'][:78,:78,0]
SC=SC[np.ix_(Ordine2021,Ordine2021)]
SC[np.where(SC == 0.)[0],np.where(SC == 0.)[1]]=9000
SC=np.log(SC)
SC[np.where(SC > 9.)[0],np.where(SC > 9.)[1]]=1e-7
SC[np.where(SC ==0)[0],np.where(SC==0)[1]]=1e-7
np.fill_diagonal(SC, 0.)


weights = 0.1*SC[np.triu_indices(nregions,1)]

# dmn=['Frontal_Sup_L','Frontal_Sup_R','Frontal_Sup_Medial_L',
#      'Frontal_Sup_Medial_R','Cingulum_Ant_L','Cingulum_Ant_R',
#      'Cingulum_Post_L','Cingulum_Post_R','ParaHippocampal_L',
#      'ParaHippocampal_R','Angular_L','Angular_R','Temporal_Mid_R','Temporal_Inf_L']

prefrontal=['Rectus_L', 'Olfactory_L', 'Frontal_Sup_Orb_L', 'Frontal_Med_Orb_L', 'Frontal_Mid_Orb_L', 'Frontal_Inf_Orb_L',
            'Rectus_R', 'Olfactory_R', 'Frontal_Sup_Orb_R', 'Frontal_Med_Orb_R', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Orb_R',]

frontal=['Frontal_Sup_L', 'Frontal_Mid_L', 'Frontal_Inf_Oper_L','Frontal_Inf_Tri_L', 'Frontal_Sup_Medial_L', 'Supp_Motor_Area_L', 'Paracentral_Lobule_L', 'Precentral_L', 'Rolandic_Oper_L', 
               'Frontal_Sup_R', 'Frontal_Mid_R', 'Frontal_Inf_Oper_R','Frontal_Inf_Tri_R', 'Frontal_Sup_Medial_R', 'Supp_Motor_Area_R', 'Paracentral_Lobule_R', 'Precentral_R', 'Rolandic_Oper_R']

occipital=['Occipital_Sup_L','Occipital_Mid_L', 'Occipital_Inf_L', 
           'Calcarine_L', 'Cuneus_L','Lingual_L', 'Fusiform_L',
          'Occipital_Sup_R','Occipital_Mid_R', 'Occipital_Inf_R', 
           'Calcarine_R', 'Cuneus_R','Lingual_R']

parietal=['Postcentral_L','Parietal_Sup_L','Parietal_Inf_L', 'SupraMarginal_L', 
          'Angular_L', 'Precuneus_L','Postcentral_R', 'Parietal_Sup_R',
         'Parietal_Inf_R', 'SupraMarginal_R', 'Angular_R', 'Precuneus_R']

temporal_pole=['Temporal_Pole_Sup_L', 'Temporal_Pole_Mid_L', 'ParaHippocampal_L','Fusiform_L',
               'Temporal_Pole_Sup_R', 'Temporal_Pole_Mid_R', 'ParaHippocampal_R','Fusiform_R']

temporal=['Heschl_L', 'Temporal_Sup_L', 'Temporal_Mid_L', 'Temporal_Inf_L', 
                    'Heschl_R', 'Temporal_Sup_R', 'Temporal_Mid_R','Temporal_Inf_R']

fon=[]
fon.extend('k' for i in range(nregions))
nodecolor=[]
nodecolor.extend('darkgray' for i in range(nregions))

# DMNlabels=np.zeros(len(dmn),dtype=np.int)
# for i in range(len(dmn)):
#     DMNlabels[i]=np.where(np.asarray(SC_labels)==dmn[i])[0][0]
#     fon[DMNlabels[i]]='k'
    
PREFRONTlabels=np.zeros(len(prefrontal),dtype=np.int)
for i in range(len(prefrontal)):
    PREFRONTlabels[i]=np.where(np.asarray(SC_labels)==prefrontal[i])[0][0]
    nodecolor[PREFRONTlabels[i]]='navy'
    
FRONTlabels=np.zeros(len(frontal),dtype=np.int)
for i in range(len(frontal)):
    FRONTlabels[i]=np.where(np.asarray(SC_labels)==frontal[i])[0][0]
    nodecolor[FRONTlabels[i]]='royalblue'

OCClabels=np.zeros(len(occipital),dtype=np.int)
for i in range(len(occipital)):
    OCClabels[i]=np.where(np.asarray(SC_labels)==occipital[i])[0][0]
    nodecolor[OCClabels[i]]='tomato'
    
PARlabels=np.zeros(len(parietal),dtype=np.int)
for i in range(len(parietal)):
    PARlabels[i]=np.where(np.asarray(SC_labels)==parietal[i])[0][0]
    nodecolor[PARlabels[i]]='#ffbf00'
    
TEMPOLElabels=np.zeros(len(temporal_pole),dtype=np.int)
for i in range(len(temporal_pole)):
    TEMPOLElabels[i]=np.where(np.asarray(SC_labels)==temporal_pole[i])[0][0]
    nodecolor[TEMPOLElabels[i]]='darkgreen'

TEMPlabels=np.zeros(len(temporal),dtype=np.int)
for i in range(len(temporal)):
    TEMPlabels[i]=np.where(np.asarray(SC_labels)==temporal[i])[0][0]
    nodecolor[TEMPlabels[i]]='limegreen'
    
#Draw curved lines
#import networkx as nx
#G = nx.Graph()
#edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
#edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G), connectionstyle='Arc3, rad=0.2')
#G = nx.DiGraph()
#G.add_edges_from(weigh)
#arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G), connectionstyle='arc3, rad=0.2')
#alphas = [0.3, 0.4, 0.5]
#for i, arc in enumerate(arcs):  # change alpha values of arcs
#    arc.set_alpha(alphas[i])