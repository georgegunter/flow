#%%
pt.figure(figsize=[20,10])
pt.plot(fpr, tpr,'-o', label='AUC: %s'%(auc_score))
pt.legend()
pt.ylabel('True positive rate')
pt.xlabel('False positive rate')
pt.title('ROC curve for low congestion weak attack')

#%%
pt.figure(figsize=[20,15])

losses = get_losses(low_test_results_medium)
labels = get_labels(low_test_results_medium)
fpr, tpr, thresholds = metrics.roc_curve(labels, losses, pos_label=1)
auc_score = metrics.auc(fpr, tpr)
pt.plot(fpr, tpr,'-o', label='Low cong. weak attack, AUC: %s'%(np.round(auc_score,2)))

losses = get_losses(low_test_results_high)
labels = get_labels(low_test_results_high)
fpr, tpr, thresholds = metrics.roc_curve(labels, losses, pos_label=1)
auc_score = metrics.auc(fpr, tpr)
pt.plot(fpr, tpr,'-o', label='Low cong. strong attack, AUC: %s'%(np.round(auc_score,2)))

losses = get_losses(med_test_results_medium)
labels = get_labels(med_test_results_medium)
fpr, tpr, thresholds = metrics.roc_curve(labels, losses, pos_label=1)
auc_score = metrics.auc(fpr, tpr)
pt.plot(fpr, tpr,'-o', label='Med cong. weak attack, AUC: %s'%(np.round(auc_score,2)))

losses = get_losses(med_test_results_high)
labels = get_labels(med_test_results_high)
fpr, tpr, thresholds = metrics.roc_curve(labels, losses, pos_label=1)
auc_score = metrics.auc(fpr, tpr)
pt.plot(fpr, tpr,'-o', label='Med cong. strong attack, AUC: %s'%(np.round(auc_score,2)))

losses = get_losses(high_test_results_medium)
labels = get_labels(high_test_results_medium)
fpr, tpr, thresholds = metrics.roc_curve(labels, losses, pos_label=1)
auc_score = metrics.auc(fpr, tpr)
pt.plot(fpr, tpr,'-o', label='High cong. weak attack, AUC: %s'%(np.round(auc_score,2)))

losses = get_losses(high_test_results_high)
labels = get_labels(high_test_results_high)
fpr, tpr, thresholds = metrics.roc_curve(labels, losses, pos_label=1)
auc_score = metrics.auc(fpr, tpr)
pt.plot(fpr, tpr,'-o', label='High cong. strong attack, AUC: %s'%(np.round(auc_score,2)))


pt.legend()
pt.ylabel('True positive rate')
pt.xlabel('False positive rate')

#%% plot cumulative losses:
    
pt.figure(figsize=[20,15])
pt.plot(np.sort(low_losses_none),np.linspace(0,1,len(low_losses_none)),'-o',label='Low cong. No Attack')
pt.plot(np.sort(low_losses_weak),np.linspace(0,1,len(low_losses_weak)),'-o',label='Low cong. Weak Attack')
pt.plot(np.sort(low_losses_strong),np.linspace(0,1,len(low_losses_strong)),'-o',label='Low cong. Strong Attack')
pt.legend()
    
#%%

pt.figure(figsize = [20,10])
pt.plot(input_vals[0],lineWidth=5)
pt.ylabel('Sample Value')

pt.figure(figsize = [20,10])
pt.plot(reconstructions[0],lineWidth=5)
pt.plot(input_vals[0],'-.',lineWidth=5)
pt.ylabel('Reconstruction Value')







