# load library
from numpy import printoptions
import pandas as pd

'''
improve methods:
1. adapt the numebr of columns included; are those numbers significant? any way to fill missing values?
'''

# read data by chunk
''' remove columns without a clear explaination, no meaningful value'''
reader = pd.read_csv('accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv', chunksize=10000)
mid=[]


# columns missing 20-50 percent;
columns_missing_20 = ['open_acc_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
       'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
       'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m']

# columns missing more than 50%; 
columns_missing_50 = ['verified_status_joint','sec_app_mths_since_last_major_derog', 'sec_app_revol_util',
       'revol_bal_joint', 'sec_app_inq_last_6mths',
       'sec_app_collections_12_mths_ex_med',
       'sec_app_chargeoff_within_12_mths', 'sec_app_num_rev_accts',
       'sec_app_open_acc', 'sec_app_mort_acc', 'sec_app_fico_range_high',
       'sec_app_fico_range_low', 'dti_joint', 'annual_inc_joint',
       'mths_since_last_record', 'mths_since_recent_bc_dlq',
       'mths_since_last_major_derog', 'mths_since_recent_revol_delinq',
       'mths_since_last_delinq']

# columns without proper explaination
cols_no_explaination = ['debt_settlement_flag', 'debt_settlement_flag_date', 'deferral_term',\
         'disbursement_method', 'hardship_amount', 'hardship_dpd', 'hardship_end_date', 'hardship_flag',\
            'hardship_last_payment_amount', 'hardship_length', 'hardship_loan_status', 'hardship_payoff_balance_amount', \
            'hardship_reason', 'hardship_start_date', 'hardship_status', 'hardship_type', 'open_act_il', \
            'orig_projected_additional_accrued_interest', 'payment_plan_start_date', 'sec_app_open_act_il',\
            'settlement_amount', 'settlement_date', 'settlement_percentage', 'settlement_status', 'settlement_term']

# drop column with high correlation
cols_high_corr = ['out_prncp_inv','funded_amnt','funded_amnt_inv','tot_hi_cred_lim','total_il_high_credit_limit']
    
# drop column with date
cols_date = ['issue_d','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d','sec_app_earliest_cr_line']

# columns that are not related to prediction
cols_not_related =  ['url','desc','id','emp_title','sub_grade','zip_code','policy_code']

# column only cotains single value
cols_single = ['member_id']

# load data
drop_cols = columns_missing_20 + columns_missing_50 + cols_no_explaination + cols_high_corr + cols_date + cols_not_related + cols_single

for chunk in reader:
    chunk.rename(columns={'verification_status_joint':'verified_status_joint'}, inplace=True)
    mid.append(chunk.drop(columns=drop_cols))
accepted = pd.concat(mid)


# loan status drop missing
accepted.dropna(subset=['loan_status'],inplace=True)
accepted['loan_status']=accepted['loan_status'].apply(lambda x: 1 if x in ['Default', 'Charged Off'] else 0).astype('object')


## int
accepted.select_dtypes('int64').columns # id should be cat value
## <M8[ns]
datetime_dict={col: [ accepted[col].unique()] for col in accepted.select_dtypes('<M8[ns]').columns}
## float
float_dict={col: [ accepted[col].unique()] for col in accepted.select_dtypes('float64').columns}
## 0bject
object_dict={col: [accepted[col].unique()] for col in accepted.select_dtypes('O').columns}

# fill missing values
## recode 
stop_words1=['just','a','the','my','need','help','to', 'one', 'me', 'for','smart', 'jc', 'low', 'looking', 'lower', 'mike\'s', 'mike',\
    'michelle\'s', 'michelle', 'many', 'needed', 'mission', 'dad\'s', 'seeking', 'high', 'it', 'new', 'nice', 'and','quick','next','level', 'more',\
        'large','small','lendingclub','better', 'me', 'you','beautiful','easy', 'finally', 'rescue', 'get','first','last','second', 'up','lower',\
            'combine','little', 'project','please', 'thank', 'thanks','ny','of','is','are','i','on','&','this','in','me,','be','with','from','-',\
                'big','short','end','our', 'needs', 'bye','two','over','will','at','some','do','clear','combine','no','or']
accepted['title'] = accepted['title'].apply( lambda x: ' '.join([s for s in x.split() if (s not in stop_words1)]) if pd.notna(x) else x )

stop_words2=['lending club','no more','-','\s','\d','$','best','better','big','bye','final','buy','finish','going','good','great','hard','high',\
    'happy','honest','less','than','responsible','rich','right','short','long','smart','smile','term','want','unexpected','the','project','new','mine',
    'profit','first','second','expense']
accepted['title'] = accepted['title'].str.lower().str.replace(' loan','').str.replace('loan ','').str.strip('!,\'-.?&#$\/\ \\0123456789%:+=\"_)(')
accepted['title'] = accepted['title'].str.replace('c.c.','credit card')
accepted['title'] = accepted['title'].str.replace('lc','lending club')
accepted['title'] = accepted['title'].str.replace('cc','credit card')

for word in stop_words2:
    accepted['title'] = accepted['title'].str.replace(word,'')

reword={'debt':['deb','debt','dedt','dbt','dept','bill','debit','wells','chase','citi','visa','bankamerica','bankofamerica','barclay','amex','american','boa','bofa','credit card',\
        'creditcard','credit','card','pay','pay-off','payoff','pay off','off','payback','paid','discover'],\
    'consolidation':['consolidation','con','cos','capitolslate','consalidadtion','conso','reconciliation','consoildation','consolidation','consolodation',\
        'consoldate','consolitation','consolidate','consol','cosolidation','onsolidation','recon','consol','refinance','refi','re-fi'],\
    'medical':['med','hospital','dental','health','surgery','dentist','rehab','headache','doctor'],\
    'wedding':['diamondring','wed','engage','honey','wedding','engagement','marr','ring'],\
    'mbuy':['mus','major','equip','daniel','defense','appliance','computer','laptop','camera','purchase','perchase','porchase',\
        'golf','boat','purchase','gun','software','seadoo','ship'],\
    'vehicle':['motor','dodge','harley','kawa','chevy','bmw','ford','toyota','wheels','honda','scooter','vehic','truck','mustang','subaru','suzuki',
        'mazda','auto','mercedes','auto','fuel','car','transmission','jeep','bike','trailer','subaru','nissan','engine','volvo','truck'],\
    'emergency':['emer','emr','emergency'],\
    'moving':['crossc','relo','moving','move'],\
    'law':['legal','attorney','law'],\
    'school':['edu','exam','training','classes','school','mba','mster\'s','student','graduate','phd','education','course','tuition','book','college',
        'teacher','program'], 
    'business':['farm','invent','business','buiness','buis','bus','bakery','shop','studio','web','buisness','company','busines','start-up',
        'startup','start up','inves'],\
    'home':['hous','heat','condo','chimney','apart','build','barn','basement','bassment','bath','boil','borrow',\
        'driveway','sewer','solar','property','cabin','yard','office','lawn','basement','renovat','home','mortgage','tub','pool','roof','rent',\
        'garage','bathroom','bedroom','kitchen','outdoor','suite','room','floor','ceil','garden','house','window','deck','fence','remodel','a/c','furniture','bed',
        'furnace','landscape','shelter','remo','tree','wash','lighttunnel'],\
    'personal':['presonal','pesonel','priv','peronal','personnel','personal','person','personal','vacation','money','cash','trip','pers','pes'],\
    'family':['child','adop','brother','sister','baby','mom','father','mother','grand','dad','daughter','kid','fam','funeral','myson'],\
    'other':['making','catch','eas','chan','fix','hop','impr','clear','com','never','add','bad','break','bright','help',\
        'bridge','sum','bless','blue','back','balan','insurance','together','all','goal','god','no','temp','self','free',\
        'live','lend','start','breath','day','clean','dream','love','peace','jan','feb','march','april','may','june','july',\
        'august','september','october','november','december','financ','redu','stres','soul','luck','month','opera','reduce',\
        'spring','time','out','plan','clos','capi','life','begin','mistake','sav','relief','air','tax','apr','interest','banks','green','vaction','simp']
        }

for k,v in reword.items():
    for item in v:
        accepted['title']=accepted['title'].apply(lambda x: (k if item in x else x) if isinstance(x,str) else x)

accepted['title'] = accepted['title'].replace({'or':'other','loan':'other','':'other','myloan':'other','future':'other','s':'other','k':'other',
'my':'other'})

top10_index = [col[0] for col in accepted[['title']].value_counts().iloc[:10].index]
accepted['title'] = accepted['title'].apply(lambda x: (x if x in top10_index else 'other') if isinstance(x,str) else x)

## fillmissing
accepted['emp_length']=accepted['emp_length'].fillna(accepted['emp_length'].mode()[0])
accepted['title']=accepted['title'].fillna(accepted['title'].mode()[0])

# unique numeric values, below 20 use mode, above 20 use median
float_dict_unique=pd.DataFrame.from_dict(float_dict, orient='index').rename(columns={0:'unique_values'})
float_dict_unique['n']=float_dict_unique['unique_values'].apply(len)
float_dict_missing = accepted[float_dict_unique.index].isnull().mean()*100
float_dict_missing = float_dict_unique.join(float_dict_missing.rename('missing percentage'))

for col in float_dict_missing.index:
    if float_dict_missing.loc[col,'n']>20:
        accepted[col].fillna(accepted[col].median(),inplace=True)
    else:
        accepted[col].fillna(accepted[col].mode()[0], inplace=True)

accepted.to_csv('./cleaned dataset.csv')
print('success')