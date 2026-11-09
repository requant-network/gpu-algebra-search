"""Exact GF(2) tensor certificates and independent flip-graph transformations."""
def bits(mask):
    while mask:
        bit=mask&-mask;yield bit.bit_length()-1;mask^=bit

def naive(shape):
    m,k,n=shape
    return [[1<<(i*k+j),1<<(j*n+l),1<<(i*n+l)] for i in range(m) for j in range(k) for l in range(n)]

def tensor(scheme,shape):
    m,k,n=shape;dimensions=(m*k,k*n,m*n);result=0
    for a,b,c in scheme:
        if any(not 0<x<1<<width for x,width in zip((a,b,c),dimensions)):raise ValueError('Invalid factor mask')
        for i in bits(a):
            for j in bits(b):result^=c<<((i*dimensions[1]+j)*dimensions[2])
    return result

def verify(scheme,shape):
    try:return tensor(scheme,shape)==tensor(naive(shape),shape)
    except (ValueError,TypeError):return False

def evaluate(scheme,a,b):
    result=0
    for x,y,z in scheme:
        if (x&a).bit_count()%2 and (y&b).bit_count()%2:result^=z
    return result

def multiply(a,b,shape):
    m,k,n=shape;result=0
    for i in range(m):
        for l in range(n):
            parity=0
            for j in range(k):parity^=((a>>(i*k+j))&1)&((b>>(j*n+l))&1)
            result|=parity<<(i*n+l)
    return result

def prune(scheme):return [list(row) for row in scheme if all(row)]

def flip(scheme,i,j,axis):
    assert i!=j and scheme[i][axis]==scheme[j][axis]
    result=[list(row) for row in scheme];v=(axis+1)%3;w=(axis+2)%3
    result[i][w]^=scheme[j][w];result[j][v]^=scheme[i][v]
    return prune(result)

def plus(scheme,i,j,axis):
    assert i!=j
    result=[list(row) for row in scheme];added=list(scheme[j]);added[axis]=scheme[i][axis]
    result[j][axis]^=scheme[i][axis];result.append(added)
    return prune(result)

def reduce_group(scheme,indices,axis,subset):
    assert len(indices)>=2 and subset>0
    assert len({scheme[i][axis] for i in indices})==1
    selected=[i for j,i in enumerate(indices) if subset>>j&1];assert len(selected)>=2
    v=(axis+1)%3;w=(axis+2)%3;dependency=0
    for i in selected:dependency^=scheme[i][v]
    assert dependency==0
    result=[list(row) for row in scheme];pivot=selected[0]
    for i in selected[1:]:result[i][w]^=scheme[pivot][w]
    result[pivot]=[0,0,0]
    return prune(result)

def kronecker_square(outer,inner,outer_side,inner_side):
    side=outer_side*inner_side;result=[]
    for a in outer:
        for b in inner:
            factors=[]
            for x,y in zip(a,b):
                mask=0
                for i in bits(x):
                    for j in bits(y):
                        row=(i//outer_side)*inner_side+j//inner_side
                        column=(i%outer_side)*inner_side+j%inner_side
                        mask|=1<<(row*side+column)
                factors.append(mask)
            result.append(factors)
    return result
