kmp算法
http://blog.csdn.net/chinwuforwork/article/details/51939826

#取出set中最长的元素
def getlongestelement(temp):
    length=0
    if not temp:#空集
        return length
    else:
        for x in temp:
            if len(x)>length:
                length=len(x)
        return length

#部分匹配表
#partial_table("ABCDABD") -> [0, 0, 0, 0, 1, 2, 0]
def partial_table(s):
    pre=set()#前缀
    post=set()#后缀
    ans=[0]

    for i in range(1,len(s)):
        pre.add(s[:i])
        post={s[j:i+1] for j in range(1,i+1)}
        ans.append( getlongestelement(pre & post ))#最长公有子串的长度
    return ans

#kmp主函数
def kmp(s,p):
    table=partial_table(p)
    m=len(s)
    n=len(p)
    ini=0#开始匹配的位置
    while ini<=m-n:
        for i in range(n):
            if s[ini+i] != p[i]:
                ini+= max(1,i-table[i-1])
                break
        else:
            return ini
    return -1
