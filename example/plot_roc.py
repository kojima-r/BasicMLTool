import matplotlib.pyplot as plt
import json

f = open('test.json', 'r')
data = json.load(f)

for filename,result in data.items():
	print(filename)
	plt.figure()
	lw = 2
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	for i,fold_result in enumerate(result['cv']):
		##多クラス問題か２クラス問題か
		if len(fold_result['prob_y'][0])!=2:
			##多クラスの場合はとりあえず最初のクラスのみ表示する
			print("fold",i,": accuracy=",fold_result['accuracy'])
			fpr, tpr, _ = fold_result['roc_curve'][0]
			roc_auc = fold_result['auc'][0]
			plt.plot(fpr, tpr,
				lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
		else:
			##2クラスの場合
			print("fold",i,": accuracy=",fold_result['accuracy'])
			fpr, tpr, _ = fold_result['roc_curve']
			roc_auc = fold_result['auc']
			plt.plot(fpr, tpr,
				lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
