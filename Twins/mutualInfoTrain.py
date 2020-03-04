import csv
import torch
from mutualInfoTrain_func import *

X, T, Y, y0, y1, Propensity = read_csv_data()
scaler, X_train_scaled, X_test_scaled, X_train, T_train, Y_train, T_test, Y_test, Propensity_train, Propensity_test = create_sets(X, T, Y, Propensity)

assert X_train_scaled.shape[0] + X_test_scaled.shape[0] == X.shape[0], "create_sets function lost examples"

model_S_learner_disableMi = calc_ATT_sLearner(X_train_scaled, X_test_scaled, T_test, Y_test, T_train, Y_train, Propensity_train, Propensity_test, enableMi=False)
model_S_learner_enableMi = calc_ATT_sLearner(X_train_scaled, X_test_scaled, T_test, Y_test, T_train, Y_train, Propensity_train, Propensity_test, enableMi=True)
print('finished train')

# predict outcomes:
treatment_vals = torch.ones_like(torch.tensor(T, dtype=torch.float).unsqueeze_(-1).cuda())
noTtreatment_vals = torch.zeros_like(torch.tensor(T, dtype=torch.float).unsqueeze_(-1).cuda())
covariates = torch.tensor(scaler.fit_transform(X), dtype=torch.float).cuda()
y1_est_disableMi, _, _ = model_S_learner_disableMi(torch.cat((covariates, treatment_vals), dim=1))
y0_est_disableMi, _, _ = model_S_learner_disableMi(torch.cat((covariates, noTtreatment_vals), dim=1))

y1_est_enableMi, _, _ = model_S_learner_enableMi(torch.cat((covariates, treatment_vals), dim=1))
y0_est_enableMi, _, _ = model_S_learner_enableMi(torch.cat((covariates, noTtreatment_vals), dim=1))

y0_est_disableMi, y1_est_disableMi = y0_est_disableMi.detach().cpu().numpy(), y1_est_disableMi.detach().cpu().numpy()
y0_est_enableMi, y1_est_enableMi = y0_est_enableMi.detach().cpu().numpy(), y1_est_enableMi.detach().cpu().numpy()

ITE_est_disableMi = (y1_est_disableMi - y0_est_disableMi)[:, 0]
ITE_est_enableMi = (y1_est_enableMi - y0_est_enableMi)[:, 0]
ITE_groundTruth = y1 - y0

ITE_est_error_disableMi = ITE_groundTruth - ITE_est_disableMi
ITE_est_error_enableMi = ITE_groundTruth - ITE_est_enableMi

plt.figure()
plt.subplot(1,2,1)
#ax=fig.add_axes([0,0,1,1])
plt.scatter(ITE_est_error_disableMi, Propensity)
#ax.scatter(grades_range, boys_grades, color='b')
plt.xlabel('ITE est error')
plt.ylabel('Propensity')
plt.title('disable')

plt.subplot(1,2,2)
#ax=fig.add_axes([0,0,1,1])
plt.scatter(ITE_est_error_enableMi, Propensity)
#ax.scatter(grades_range, boys_grades, color='b')
plt.xlabel('ITE est error')
plt.ylabel('Propensity')
plt.title('enable')

methodsErrDiffPerPatient = np.abs(ITE_est_error_disableMi) - np.abs(ITE_est_error_enableMi)
n_bins = 100
plt.figure()
n, bins, patches = plt.hist(methodsErrDiffPerPatient, n_bins, density=True, histtype='step', cumulative=True, label='Methods Err Diff')
#n, bins, patches = plt.hist(ITE_onThoseWithPositiveOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_pos_est_error')
#n, bins, patches = plt.hist(ITE_onThoseWithNegativeOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_neg_est_error')
plt.legend()
plt.grid(True)
plt.title('ITE estimation method error diff hist: abs(disable)-abs(enable)')
#plt.savefig('ITE_estimation_error_hist')
#plt.show()

positiveOutcomeIndexes = np.where(Y == 1)[0]
negativeOutcomeIndexes = np.where(Y == 0)[0]
positiveIteIndexes = np.where(ITE_groundTruth == 1)[0]
negativeIteIndexes = np.where(ITE_groundTruth == -1)[0]
neutralIteIndexes = np.where(ITE_groundTruth == 0)[0]
LightBabyDieIndexes = np.where(y0 == 1)[0]
LightBabyLiveIndexes = np.where(y0 == 0)[0]
neutralIteBabyDiesIndexes = np.intersect1d(neutralIteIndexes, LightBabyDieIndexes)
neutralIteBabyLivesIndexes = np.intersect1d(neutralIteIndexes, LightBabyLiveIndexes)

print(f'% positive (death) outcomes: {100*positiveOutcomeIndexes.shape[0]/Y.shape[0]}; % negative (death) outcomes: {100*negativeOutcomeIndexes.shape[0]/Y.shape[0]}')

print(f'% HeavyBabiesInDanger: {100*positiveIteIndexes.shape[0]/Y.shape[0]}; % LightBabiesInDanger: {100*negativeIteIndexes.shape[0]/Y.shape[0]};  % neutralIte: {100*neutralIteIndexes.shape[0]/Y.shape[0]} out of which {100*y0[neutralIteIndexes].sum()/y0[neutralIteIndexes].shape[0]} % will die')

ITE_onThoseWithPositiveOutcome_est_error_disableMi = ITE_groundTruth[positiveOutcomeIndexes]- ITE_est_disableMi[positiveOutcomeIndexes]
ITE_onThoseWithNegativeOutcome_est_error_disableMi = ITE_groundTruth[negativeOutcomeIndexes]- ITE_est_disableMi[negativeOutcomeIndexes]

ITE_onThoseWithPositiveOutcome_est_error_enableMi = ITE_groundTruth[positiveOutcomeIndexes]- ITE_est_enableMi[positiveOutcomeIndexes]
ITE_onThoseWithNegativeOutcome_est_error_enableMi = ITE_groundTruth[negativeOutcomeIndexes]- ITE_est_enableMi[negativeOutcomeIndexes]

ITE_onThoseWithPositiveIte_est_error_disableMi = ITE_groundTruth[positiveIteIndexes]- ITE_est_disableMi[positiveIteIndexes]
ITE_onThoseWithNegativeIte_est_error_disableMi = ITE_groundTruth[negativeIteIndexes]- ITE_est_disableMi[negativeIteIndexes]
ITE_onThoseWithNeutralIte_est_error_disableMi = ITE_groundTruth[neutralIteIndexes]- ITE_est_disableMi[neutralIteIndexes]

ITE_onThoseWithPositiveIte_est_error_enableMi = ITE_groundTruth[positiveIteIndexes]- ITE_est_enableMi[positiveIteIndexes]
ITE_onThoseWithNegativeIte_est_error_enableMi = ITE_groundTruth[negativeIteIndexes]- ITE_est_enableMi[negativeIteIndexes]
ITE_onThoseWithNeutralIte_est_error_enableMi = ITE_groundTruth[neutralIteIndexes]- ITE_est_enableMi[neutralIteIndexes]
print('')
print('Group of babies in which the light is in danger:')
disableMiEstOutcomeOfLightBabiesInDanger = y0_est_disableMi[negativeIteIndexes]
enableMiEstOutcomeOfLightBabiesInDanger = y0_est_enableMi[negativeIteIndexes]
print(f'    Light babies in danger: disableMi predicts that {100*disableMiEstOutcomeOfLightBabiesInDanger.round().sum()/disableMiEstOutcomeOfLightBabiesInDanger.shape[0]} % will die (when in fact all die)')
print(f'    Light babies in danger: enableMi predicts that {100*enableMiEstOutcomeOfLightBabiesInDanger.round().sum()/enableMiEstOutcomeOfLightBabiesInDanger.shape[0]} % will die (when in fact all die)')

print('')
disableMiEstOutcomeOfHeavyBabiesSafe = y1_est_disableMi[negativeIteIndexes]
enableMiEstOutcomeOfHeavyBabiesSafe = y1_est_enableMi[negativeIteIndexes]
print(f'    Heavy babies safe: disableMi predicts that {100*disableMiEstOutcomeOfHeavyBabiesSafe.round().sum()/disableMiEstOutcomeOfHeavyBabiesSafe.shape[0]} % will die (when in fact all live)')
print(f'    Heavy babies safe: enableMi predicts that {100*enableMiEstOutcomeOfHeavyBabiesSafe.round().sum()/enableMiEstOutcomeOfHeavyBabiesSafe.shape[0]} % will die (when in fact all live)')

print('')
print('Group of babies in which the heavy is in danger:')
disableMiEstOutcomeOfHeavyBabiesInDanger = y1_est_disableMi[positiveIteIndexes]
enableMiEstOutcomeOfHeavyBabiesInDanger = y1_est_enableMi[positiveIteIndexes]
print(f'    Heavy babies in danger: disableMi predicts that {100*disableMiEstOutcomeOfHeavyBabiesInDanger.round().sum()/disableMiEstOutcomeOfHeavyBabiesInDanger.shape[0]} % will die (when in fact all die)')
print(f'    Heavy babies in danger: enableMi predicts that {100*enableMiEstOutcomeOfHeavyBabiesInDanger.round().sum()/enableMiEstOutcomeOfHeavyBabiesInDanger.shape[0]} % will die (when in fact all die)')

print('')
disableMiEstOutcomeOfLightBabiesSafe = y0_est_disableMi[positiveIteIndexes]
enableMiEstOutcomeOfLightBabiesSafe = y0_est_enableMi[positiveIteIndexes]
print(f'    Light babies safe: disableMi predicts that {100*disableMiEstOutcomeOfLightBabiesSafe.round().sum()/disableMiEstOutcomeOfLightBabiesSafe.shape[0]} % will die (when in fact all live)')
print(f'    Light babies safe: enableMi predicts that {100*enableMiEstOutcomeOfLightBabiesSafe.round().sum()/enableMiEstOutcomeOfLightBabiesSafe.shape[0]} % will die (when in fact all live)')

print('')
print('Group of babies in which both are the same:')
disableMiEstOutcomeOfNeutralBabiesInDanger = np.concatenate((y1_est_disableMi[neutralIteBabyDiesIndexes], y0_est_disableMi[neutralIteBabyDiesIndexes]), axis=0)
enableMiEstOutcomeOfNeutralBabiesInDanger = np.concatenate((y1_est_enableMi[neutralIteBabyDiesIndexes], y0_est_enableMi[neutralIteBabyDiesIndexes]), axis=0)
print(f'    Neutral babies in danger: disableMi predicts that {100*disableMiEstOutcomeOfNeutralBabiesInDanger.round().sum()/disableMiEstOutcomeOfNeutralBabiesInDanger.shape[0]} % will die (when in fact all die)')
print(f'    Neutral babies in danger: enableMi predicts that {100*enableMiEstOutcomeOfNeutralBabiesInDanger.round().sum()/enableMiEstOutcomeOfNeutralBabiesInDanger.shape[0]} % will die (when in fact all die)')

print('')
disableMiEstOutcomeOfNeutralBabiesSafe = np.concatenate((y1_est_disableMi[neutralIteBabyLivesIndexes], y0_est_disableMi[neutralIteBabyLivesIndexes]), axis=0)
enableMiEstOutcomeOfNeutralBabiesSafe = np.concatenate((y1_est_enableMi[neutralIteBabyLivesIndexes], y0_est_enableMi[neutralIteBabyLivesIndexes]), axis=0)
print(f'    Neutral babies safe: disableMi predicts that {100*disableMiEstOutcomeOfNeutralBabiesSafe.round().sum()/disableMiEstOutcomeOfNeutralBabiesSafe.shape[0]} % will die (when in fact all live)')
print(f'    Neutral babies safe: enableMi predicts that {100*enableMiEstOutcomeOfNeutralBabiesSafe.round().sum()/enableMiEstOutcomeOfNeutralBabiesSafe.shape[0]} % will die (when in fact all live)')

n_bins = 5
plt.figure()
n, bins, patches = plt.hist([ITE_est_error_disableMi, ITE_onThoseWithPositiveOutcome_est_error_disableMi, ITE_onThoseWithNegativeOutcome_est_error_disableMi], n_bins, density=True, histtype='step', cumulative=False, label=['ITE_est_error_disableMi', 'ite_pos_est_error', 'ite_neg_est_error'])
#n, bins, patches = plt.hist(ITE_onThoseWithPositiveOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_pos_est_error')
#n, bins, patches = plt.hist(ITE_onThoseWithNegativeOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_neg_est_error')
plt.legend()
plt.grid(True)
plt.title('ITE estimation (disable) error hists; deathOutErr=%0.2f; liveOutErr=%0.2f' % (ITE_onThoseWithPositiveOutcome_est_error_disableMi.__abs__().mean(), ITE_onThoseWithNegativeOutcome_est_error_disableMi.__abs__().mean()))
plt.savefig('ITE_estimation_error_hist')

n_bins = 5
plt.figure()
n, bins, patches = plt.hist([ITE_est_error_enableMi, ITE_onThoseWithPositiveOutcome_est_error_enableMi, ITE_onThoseWithNegativeOutcome_est_error_enableMi], n_bins, density=True, histtype='step', cumulative=False, label=['ITE_est_error_enableMi', 'ite_pos_est_error', 'ite_neg_est_error'])
#n, bins, patches = plt.hist(ITE_onThoseWithPositiveOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_pos_est_error')
#n, bins, patches = plt.hist(ITE_onThoseWithNegativeOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_neg_est_error')
plt.legend()
plt.grid(True)
plt.title('ITE estimation (enable) error hists; deathOutErr=%0.2f; liveOutErr=%0.2f' % (ITE_onThoseWithPositiveOutcome_est_error_enableMi.__abs__().mean(), ITE_onThoseWithNegativeOutcome_est_error_enableMi.__abs__().mean()))
plt.savefig('ITE_estimation_error_hist')

n_bins = 5
plt.figure()
n, bins, patches = plt.hist([ITE_est_error_disableMi, ITE_onThoseWithPositiveIte_est_error_disableMi, ITE_onThoseWithNegativeIte_est_error_disableMi, ITE_onThoseWithNeutralIte_est_error_disableMi], n_bins, density=True, histtype='step', cumulative=False, label=['ITE_est_error_disableMi', 'ite_posIte_error', 'ite_negIte_error', 'ite_neuIte_error'])
#n, bins, patches = plt.hist(ITE_onThoseWithPositiveOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_pos_est_error')
#n, bins, patches = plt.hist(ITE_onThoseWithNegativeOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_neg_est_error')
plt.legend()
plt.grid(True)
plt.title('ITE estimation (disable) error hists; posIteErr=%0.2f; negIteErr=%0.2f, neuIteErr=%0.2f' % (ITE_onThoseWithPositiveIte_est_error_disableMi.__abs__().mean(), ITE_onThoseWithNegativeIte_est_error_disableMi.__abs__().mean(), ITE_onThoseWithNeutralIte_est_error_disableMi.__abs__().mean()))
plt.savefig('ITE_estimation_error_hist')

n_bins = 5
plt.figure()
n, bins, patches = plt.hist([ITE_est_error_enableMi, ITE_onThoseWithPositiveIte_est_error_enableMi, ITE_onThoseWithNegativeIte_est_error_enableMi, ITE_onThoseWithNeutralIte_est_error_enableMi], n_bins, density=True, histtype='step', cumulative=False, label=['ITE_est_error_enableMi', 'ite_posIte_error', 'ite_negIte_error', 'ite_neuIte_error'])
#n, bins, patches = plt.hist(ITE_onThoseWithPositiveOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_pos_est_error')
#n, bins, patches = plt.hist(ITE_onThoseWithNegativeOutcome_est_error_disableMi, n_bins, density=True, histtype='step', cumulative=False, label='ite_neg_est_error')
plt.legend()
plt.grid(True)
plt.title('ITE estimation (enable) error hists; posIteErr=%0.2f; negIteErr=%0.2f, neuIteErr=%0.2f' % (ITE_onThoseWithPositiveIte_est_error_enableMi.__abs__().mean(), ITE_onThoseWithNegativeIte_est_error_enableMi.__abs__().mean(), ITE_onThoseWithNeutralIte_est_error_enableMi.__abs__().mean()))
plt.savefig('ITE_estimation_error_hist')
plt.show()

# negative-ITE: y1-y0<0 ==> y1<y0 <==> be the small baby is more dangerous
# In the cases in which the small baby dies the new method is better

# positive-ITE: y1-y0>0 ==> y1>y0 <==> be the big baby is more dangerous