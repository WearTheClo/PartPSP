import math

def topo_read(topo_path):
    topo=[]
    with open(topo_path,"r") as f:
        network=f.read().splitlines()
    for line in network:
        line =line.split()
        tem=[]
        for i in line:
            tem.append(float(i))
        topo.append(tem)
    return topo

def LocalUpdate(topo,B,world_size):
    for i in range(B-1):
        for j in range(world_size):
            tem=[0.0]*world_size
            tem[j]=1.0
            topo.append(tem)

    if B*world_size != len(topo):
        raise ValueError("length of topo not match with N*B")

    return topo

def Exponential(world_size):
    B = int(math.log(world_size-1,2))+1
    topo=[]
    for i in range(B):
        for j in range(world_size):
            tem=[0.0]*world_size
            tem[j]=1.0
            step=int(math.pow(2,i))
            idx=(j+step)%world_size
            tem[idx]=1.0
            topo.append(tem)
    return topo


def Generate(outdegree, world_size):
    topo=[]
    for i in range(world_size):
        tem=[0.0]*world_size
        for j in range(outdegree):
            tem[(i+j)%world_size] = 1.0
        topo.append(tem)
    return topo

def TopoLoador(args):
    if args.strategy=='Static':
        topo=topo_read(args.topology)
        return LocalUpdate(topo,args.B,args.world_size)
    if args.strategy=='Exponential':
        return Exponential(args.world_size)
    if args.strategy=='Generate':
        return Generate(args.outdegree, args.world_size)
    return topo
