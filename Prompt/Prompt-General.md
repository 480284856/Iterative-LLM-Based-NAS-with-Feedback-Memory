实现整个pipeline。这个pipeline是要根据训练反馈不停的修改prompt，直到代码的准确率达到要求。

整个pipeline的流程如下：
1. 使用CodeGenerator生成初始代码。
2. 使用CodeExtractor提取代码中的代码。
3. 使用Evaluator评估提取的代码的准确率。
4. 如果evaluate成功，则记录准确率+pipeline的迭代轮数到一个文件里。（注意，这个文件会被多次写入，所以需要每次都追加写入）
5. 使用PromptImprover改进提示词。
6. 重复1-5步骤，直到准确率达到要求。

Code Generator：
我给出了CodeGenerator.md，这个是其初始的prompt。
你需要自行实现Code Generator。

Evaluator：
我给出了Evaluator.md，你可以参考这个实现Evaluator。

Prompt Improver：
我给出了Prompt Improver.md，这个是其初始的prompt。
你需要自行实现Prompt Improver。