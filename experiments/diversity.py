"""Select best distinct certificates without counting term permutations twice."""
def elites(schemes, ranks, count=8):
    assert len(schemes)==len(ranks) and len(schemes)>0 and count>0
    seen=set();selected=[]
    for index in sorted(range(len(ranks)),key=lambda i:(ranks[i],i)):
        canonical=tuple(sorted(tuple(map(int,row)) for row in schemes[index] if row[0]))
        assert len(canonical)==ranks[index]
        if canonical not in seen:
            seen.add(canonical);selected.append(index)
            if len(selected)==count:break
    assert selected
    return selected


if __name__=='__main__':
    a=[[1,2,3],[4,5,6]];b=list(reversed(a));c=[[7,8,9]]
    assert elites([a,b,c],[2,2,1])==[2,0]
    assert elites([a,b],[2,2])==[0]
    assert elites([a,c],[2,1],1)==[1]
    assert a==[[1,2,3],[4,5,6]]
    print('Diverse elite selection, permutation deduplication and nonmutation pass')
