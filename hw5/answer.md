# 体系结构作业 5

[TOC]

## 1

假定处理器运⾏频率为 700MHZ，最⼤向量⻓度为 64，载⼊/存储单元的启动开销为 15 个时钟周期，乘法单元为 8 个时 钟周期，加法/减法单元为 5 个时钟周期。在该处理器上进⾏如下运算，将两个包含单精度复数值的向量相乘:

```cpp
for (i = 0; i < 300; i++) 
{
	c_re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i];
	c_im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
}
```

 

问题：

### 这个内核的运算密度为多少 (注:运算密度指运⾏程序时执⾏的浮点运算数除以主存储器中访问的字节数)? 

每6个浮点计算，从主存读取4个浮点数，写入2个浮点数。从而强度是：
$$
\frac{6}{(4+2)*4}=0.25
$$

### 将此循环转换为使⽤条带挖掘(Strip Mining)的 VMIPS 汇编代码。 

```assembly
		li $VL,44					# 300-256=44
		li $r1,0					# 0存入$r1，初始化索引
loop:	lv $v1,a_re+$r1				# 取a_re
		lv $v3,b_re+$r1 			# 取b_re
		mulvv.s $v5,$v1,$v3 		# a_re*b_re
		lv $v2,a_im+$r1				# 取a_im
		lv $v4,b_im+$r1 			# 取b_im
		mulvv.s $v6,$v2,$v4 		# a_im*b_im
		subvv.s $v5,$v5,$v6 		# a_re*b_re - a_im*b_im
		sv $v5,c_re+$r1 			# 存c_re
		mulvv.s $v5,$v1,$v4			# a_re*b_im
		mulvv.s $v6,$v2,$v3 		# a_im*b_re
		addvv.s $v5,$v5,$v6 		# a_re*b_im + a_im*b_re
		sv $v5,c_im+$r1				# 存c_im
		bne $r1,0,else				# 检查是否是第一次迭代
        addi $r1,$r1,#176            # 第一次迭代 增加44*4=176
        j loop                      # 保证下一次迭代执行
else:   addi $r1,$r1,256            # 不是第一次迭代 自增256
skip:   blt  $r1,1200,loop          # 继续迭代？ 300*4=1200

```



### 假定采⽤链接和单⼀存储器流⽔线，需要多少次钟鸣?每个复数结果值需要多少个时钟周期(包括启动开销在内)? 

```assembly
lv #load a_re
lv #load b_re
mulvv.s lv # a_re*b_re, load a_im
lv mulvv.s #load b_im, a_im*b_im
subvv.s sv #a_re*b_re-a_im*b_im, store c_re
mulvv.s # a_re*b_im
mulvv.s # a_im*b_re
addvv.s sv # a_re*b_im+a_im*b_re, store c_im
```

需要
$$
\text{ceil}(\frac{300}{64})*6=30
$$
30次钟鸣

每一个复数结果值需要的时钟周期是：
$$
\frac{6*64+15*6+8*4+5*2}{2*64}=\frac{129}{32}
$$

### 现在假定处理器有三条存储器流⽔线和链接。如果该循环的访问过程中没有组冲突， 每个结果需要多少个时钟周期?

```assembly
mulvv.s # a_re * b_re
mulvv.s # a_im * b_im
subvv.s sv # a_re * b_re-a_im * b_im, store c_re
mulvv.s # a_re * b_im
mulvv.s lv # a_im * b_re, load the next a_re
addvv.s sv lv lv lv # a_re * b_im + a_im * b_re, store c_im, load the next b_re,a_im,b_im
```

钟鸣次数还是
$$
\text{ceil}(\frac{300}{64})*6=30
$$
则钟鸣次数不变，结果不变

还是
$$
\frac{6*64+15*6+8*4+5*2}{2*64}=\frac{129}{32}
$$

## 2

### 2.1

$$
1.5\text{GHz}\times 16\times 16 = 384 \text{GFLOP/s}
$$

### 2.2

每一个单精度运算读两个操作数 写入一个操作数 访问12个字节 从而需要
$$
12\text{Byte}\times 384\text{GFLOP/s}=4608\text{GB/s}
$$
因此不可持续。

