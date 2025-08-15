# 一点高能物理+监督学习

- 用监督学习的方法去分离dijet中的本地和信号
- 2025年iSTEP暑期学校
- 个人学习笔记，数据以及部分代码来自于CERN和暑期学校课程材料

---

- 集成了一点函数用来实现训练，需要时可以直接调用
- 一般是在 ex_2_copy.ipynb 中运行即可
- 如果需要绘制ROC曲线，使用 draw.ipynb
- 若期望fpr和tpr等数据，使用 draw_modified.py
- 训练好的模型的参数存储在 improved_model/, simple_mlp_zprime/, simple_mlp_zprime_new_cos_anl/中，可以直接使用（注意，有些模型因为各种原因临时中断了训练，可能不会有ckpt文件）

*特此声明*
- iSTEP_SupervisedLearning_Exercise.ipynb文件是暑期学校提供的代码，我做了小部分修改
- 特此感谢暑校里三位教python的老师，感谢他们的耐心讲解和帮助

---

- 代码中的注释仅供参考
- 编写过程中使用了DeepSeek和CodeGeeX，感谢他们的帮助
- 感谢CERN和暑期学校课程材料，感谢各位老师的辛勤付出，感谢各位同学的讨论与帮助
- 本次暑期学校对我而言是一次非常宝贵的经历，感谢各位老师和同学们的陪伴