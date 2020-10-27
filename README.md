# UnsuperPoint_PyTorch
A pytorch version about unsuper point

# 致谢
论文：https://arxiv.org/abs/1907.04011

@RPFey实现：https://github.com/RPFey/UnsuperPoint (基本功能都完备了)

# 说明
代码实现的是UnsuperPoint的pytorch版本，参考之前@RPFey的实现，修改了一部分不能训练成功的地方，在slam数据集中实际测试可以达到orb的效果。
代码中的实现和论文的内容不完全一致，论文使用描述子各个bit的相关系数来监督描述子的表达能力，代码直接使用不同位置描述子的特异性来监督。

***更多的验证数据集和验证方法，请自行增加。更合理，高精度的方法请自行修改***

# 复现笔记
有道云：http://note.youdao.com/s/IQBrgPio
