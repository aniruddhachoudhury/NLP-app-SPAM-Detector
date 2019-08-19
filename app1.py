from flask import Flask,render_template
from flask import request
import pandas as pd 
import pyhdb
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams["figure.figsize"]=12,6
import warnings
warnings.filterwarnings('ignore')
from fbprophet import Prophet
import logging
logging = logging.getLogger()

#----------------------------------------------------------------------------------


holidays=pd.read_csv("holidays.csv")


def pred_working_hours_day(Prediction_Days,Employee_IDs,dataset):
    Employee_Day_Working_Hours_Prediction=pd.DataFrame(columns=["EmpID","ds","yhat"])
    Employee_Day_Working_Hours_Actual_Prediction=pd.DataFrame(columns=["EmpID","ds","yhat"])
    Pred_Days=int(Prediction_Days)
    IDs=Employee_IDs
    for ID in IDs:
        Employee_day_dataset=dataset[dataset["EmpID"]==ID]
        Employee_day_dataset = Employee_day_dataset.T.reset_index()
        col = Employee_day_dataset.columns[1]
        Employee_day_dataset.rename(columns={'index':'ds',col:'y'},inplace=True)
        Employee_day_dataset.drop([0],inplace=True,axis=0)
        Employee_day_dataset.fillna(0,inplace=True)
        Employee_day_dataset['ds']=pd.to_datetime(Employee_day_dataset["ds"])
        Date=Employee_day_dataset["ds"].max()
        model_days = Prophet(interval_width = 0.95,holidays=holidays)
        model_days.fit(Employee_day_dataset)
        future_days= model_days.make_future_dataframe(periods=Pred_Days,include_history=True,freq="D")
        forecast_days = model_days.predict(future_days)
        forecast_days=pd.DataFrame(forecast_days)
        forecast_days["EmpID"]=ID
        Employee_Day_Working_Hours_Prediction=pd.concat([Employee_Day_Working_Hours_Prediction,forecast_days[forecast_days["ds"]>Date]],join="inner",axis=0,sort=True)
        Employee_Day_Working_Hours_Prediction["yhat"]=Employee_Day_Working_Hours_Prediction["yhat"].apply(np.floor)
        Employee_Day_Working_Hours_Actual_Prediction=pd.concat([Employee_Day_Working_Hours_Actual_Prediction,forecast_days[forecast_days["ds"]<=Date]],join="inner",axis=0,sort=False)
        Employee_Day_Working_Hours_Actual_Prediction["Actual_Working_Hours"]=Employee_day_dataset["y"]
    Employee_Day_Working_Hours_Prediction.rename(columns={"ds":"Date","yhat":"Predicted Working Hours"},inplace=True)
    Employee_Day_Working_Hours_Actual_Prediction.rename(columns={"ds":"Date","yhat":"Predicted Working Hours"},inplace=True)
    return Employee_Day_Working_Hours_Prediction,Employee_Day_Working_Hours_Actual_Prediction
def pred_working_hours_week(Prediction_Days,Employee_IDs,dataset):
    Employee_Week_Working_Hours_Prediction=pd.DataFrame(columns=["EmpID","ds","yhat"])
    Employee_Week_Working_Hours_Actual_Prediction=pd.DataFrame(columns=["EmpID","ds","yhat"])
    Pred_Days=Prediction_Days
    IDs=Employee_IDs
    for ID in IDs:
        Employee_dataset=dataset[dataset["EmpID"]==ID]
        Employee_dataset = Employee_dataset.T.reset_index()
        col = Employee_dataset.columns[1]
        Employee_dataset.rename(columns={'index':'ds',col:'y'},inplace=True)
        Employee_dataset.drop([0],inplace=True,axis=0)
        Employee_dataset.fillna(0,inplace=True)
        Employee_dataset["ds"]=pd.to_datetime(Employee_dataset["ds"])
        Date=Employee_dataset["ds"].max()
        indexeddataset=Employee_dataset.set_index(["ds"])
        indexeddataset=indexeddataset.resample("W").sum()
        indexeddataset["ds"]=indexeddataset.index
        model_week = Prophet(interval_width = 0.95,holidays=holidays)
        model_week.fit(indexeddataset)
        future_weeks= model_week.make_future_dataframe(periods=int(Pred_Days//7),include_history=True,freq="W")
        forecast_weeks = model_week.predict(future_weeks)
        forecast_weeks=pd.DataFrame(forecast_weeks)
        forecast_weeks["EmpID"]=ID
        Employee_Week_Working_Hours_Prediction=pd.concat([Employee_Week_Working_Hours_Prediction,forecast_weeks[forecast_weeks["ds"]>Date]],join="inner",axis=0,sort=True)
        Employee_Week_Working_Hours_Prediction["yhat"]=Employee_Week_Working_Hours_Prediction["yhat"].apply(np.floor)
        Employee_Week_Working_Hours_Actual_Prediction=pd.concat([Employee_Week_Working_Hours_Actual_Prediction,forecast_weeks[forecast_weeks["ds"]<=Date]],join="inner",axis=0,sort=False)
        Employee_Week_Working_Hours_Actual_Prediction["Actual Working Hours"]=indexeddataset["y"]
    Employee_Week_Working_Hours_Prediction.rename(columns={"ds":"Date","yhat":"Predicted Working Hours"},inplace=True)
    Employee_Week_Working_Hours_Actual_Prediction.rename(columns={"ds":"Date","yhat":"Predicted Working Hours"},inplace=True)
    return Employee_Week_Working_Hours_Prediction,Employee_Week_Working_Hours_Actual_Prediction
def pred_week_leave(Prediction_Days,Employee_IDs,dataset):
    Employee_Leave_Prediction=pd.DataFrame(columns=["EmpID","ds","yhat"])
    Employee_Leave_Actual_Prediction=pd.DataFrame(columns=["EmpID","ds","yhat"])
    Pred_Days=Prediction_Days
    IDs=Employee_IDs
    for ID in IDs:
        Employee_dataset = dataset[dataset['EmpID']==ID]
        Employee_dataset = Employee_dataset.T.reset_index()
        col = Employee_dataset.columns[1]
        Employee_dataset.rename(columns={'index':'ds',col:'y1'},inplace=True)
        Employee_dataset.drop([0],inplace=True,axis=0)
        Employee_dataset.fillna(0,inplace=True)
        Employee_dataset["y1"]=Employee_dataset["y1"].astype(int)
        Employee_dataset['y'] =pd.Series(Employee_dataset['y1'] == 0)
        Employee_dataset['y'].replace(True,1,inplace=True)
        Employee_dataset['y'].replace(False,0,inplace=True)
        Employee_dataset.drop("y1",inplace=True,axis=1)
        Employee_dataset["ds"]=pd.to_datetime(Employee_dataset["ds"])
        Date=Employee_dataset["ds"].max()
        indexeddataset=Employee_dataset.set_index(["ds"])
        indexeddataset=indexeddataset.resample("W").sum()
        y=[]
        for val in indexeddataset["y"]:
            if val>=2:
                y.append(val-2)
            elif val>0 and val<2:
                y.append(val-1)
            else:
                y.append(val)
        indexeddataset["y"]=y
        indexeddataset["ds"]=indexeddataset.index
        model_leave = Prophet(interval_width = 0.95)
        model_leave.fit(indexeddataset)
        future_leaves= model_leave.make_future_dataframe(periods=int(Pred_Days//7),include_history=True,freq="w")
        forecast_leaves = model_leave.predict(future_leaves)
        forecast_leaves=pd.DataFrame(forecast_leaves)
        forecast_leaves["EmpID"]=ID
        Employee_Leave_Prediction=pd.concat([Employee_Leave_Prediction,forecast_leaves[forecast_leaves["ds"]>Date]],join="inner",axis=0,sort=True)
        Employee_Leave_Prediction["yhat"]=Employee_Leave_Prediction["yhat"].apply(np.round)
        Employee_Leave_Actual_Prediction=pd.concat([Employee_Leave_Actual_Prediction,forecast_leaves[forecast_leaves["ds"]<=Date]],join="inner",axis=0,sort=False)
        Employee_Leave_Actual_Prediction["Actual Leaves"]=indexeddataset["y"]
    Employee_Leave_Prediction.rename(columns={"ds":"Date","yhat":"Predicted Leave"},inplace=True)
    Employee_Leave_Actual_Prediction.rename(columns={"ds":"Date","yhat":"Predicted Leave"},inplace=True)
    return Employee_Leave_Prediction,Employee_Leave_Actual_Prediction


# In[28]:


def master_program (Project_Duration,Employee_IDs):    
    dataset=pd.read_csv("merged_file.csv")
    Prediction_Days=int(Project_Duration)+(int(Project_Duration)//22)*8
    Number_Of_Employees=len(Employee_IDs)
    No_Of_Days=int(Prediction_Days)-(int(Prediction_Days)//30)*8

    day_working_hours_pred,day_working_hours_actual_pred=pred_working_hours_day(Prediction_Days,Employee_IDs,dataset)
    week_working_hours_pred,week_working_hours_actual_pred=pred_working_hours_week(Prediction_Days,Employee_IDs,dataset)
    week_leave_pred,week_leave_actual_pred=pred_week_leave(Prediction_Days,Employee_IDs,dataset)


    day_working_hours_pred["DayOfWeek"]=day_working_hours_pred["Date"].dt.dayofweek
    day_working_hours_pred=day_working_hours_pred[day_working_hours_pred["DayOfWeek"]<5]
    day_working_hours_pred.drop("DayOfWeek",axis=1,inplace=True)

    day_working_hours_actual_pred["DayOfWeek"]=day_working_hours_actual_pred["Date"].dt.dayofweek
    day_working_hours_actual_pred=day_working_hours_actual_pred[day_working_hours_actual_pred["DayOfWeek"]<5]
    day_working_hours_actual_pred.drop("DayOfWeek",axis=1,inplace=True)
    day_working_hours_actual_pred["Predicted Working Hours"]=day_working_hours_actual_pred["Predicted Working Hours"].apply(np.ceil)
    day_working_hours_actual_pred["Actual_Working_Hours"]=day_working_hours_actual_pred["Actual_Working_Hours"].apply(np.ceil)

    day_working_hours_pred.to_csv("day_working_hours_pred.csv",index=False)
    day_working_hours_actual_pred.to_csv("day_working_hours_actual_pred.csv",index=False)
    week_working_hours_pred.to_csv("week_working_hours_pred.csv",index=False)
    week_leave_pred.to_csv("week_leave_pred.csv",index=False)


    Total_Project_Hours=int(No_Of_Days)*9*Number_Of_Employees
    #print ('Total project hours',Total_Project_Hours)

    Daily_Estimated_Working_Hours=day_working_hours_pred["Predicted Working Hours"].groupby(day_working_hours_pred["EmpID"]).sum().reset_index()
    Daily_Working_Hours=Daily_Estimated_Working_Hours["Predicted Working Hours"].sum()
    Weekly_Estimated_Working_Hours=week_working_hours_pred["Predicted Working Hours"].groupby(week_working_hours_pred["EmpID"]).sum().reset_index()
    Weekly_Working_Hours=Weekly_Estimated_Working_Hours["Predicted Working Hours"].sum()
    Weekly_Estimated_Leave=week_leave_pred["Predicted Leave"].groupby(week_leave_pred["EmpID"]).sum().reset_index()
    Weekly_Leaves=Weekly_Estimated_Leave["Predicted Leave"].sum()
    Project_Prediction_summary=pd.DataFrame(columns=["Total Project Duration(days)","Delay In Project(days)","EmpID","Reason1","Reason2"])

    Delay_Duration=Total_Project_Hours-Daily_Working_Hours
    """if Delay_Duration < 0:
        print("Project Will be Completed before {} hrs".format(abs(Delay_Duration)))
    else:
        print("Project Will Be delayed by {} hrs".format(Delay_Duration))
   """


    #for ID in Daily_Estimated_Working_Hours["EmpID"]:

    Total_Gap_In_Hrs=0
    for ID in Daily_Estimated_Working_Hours["EmpID"]:
        Hours= Daily_Estimated_Working_Hours[Daily_Estimated_Working_Hours["EmpID"]==ID]["Predicted Working Hours"].sum() 
        Leaves =Weekly_Estimated_Leave[Weekly_Estimated_Leave["EmpID"]==ID]["Predicted Leave"].sum()
        Gap_In_Hrs=(No_Of_Days-Leaves)*9-Hours
        #print(Gap_In_Hrs)
        if Gap_In_Hrs> 0:
            Total_Gap_In_Hrs+=Gap_In_Hrs
        d={}
        d["EmpID"]=ID

        if Leaves > 0:
           # print("Reason : {} might take leave for {} days".format(ID,Leaves))
            d["Reason1"]="might take Leave for"+" "+str(Leaves)+" "+"days"
        if Gap_In_Hrs > 0:
            #print("Reason : {} might be short by {} hrs in {} days".format(ID,Gap_In_Hrs,Project_Duration))
            d["Reason2"]="might be short by"+" "+str(Gap_In_Hrs)+" "+"Hrs"
        Project_Prediction_summary=pd.concat([Project_Prediction_summary,pd.DataFrame([d])],axis=0,sort=False,ignore_index=True)
    Project_Prediction_summary.iloc[0,0]=Project_Duration
    Project_Prediction_summary.iloc[0,1]=week_leave_pred["Predicted Leave"].sum()+int(Total_Gap_In_Hrs//9)
    Project_Prediction_summary.to_csv("Project_Prediction_summary.csv")


    connection = pyhdb.connect(
        host="localhost",
        port=30015,
        user="DEVUSER",
        password="Ih8p@sswords"
    )   
    cursor = connection.cursor()
    #----------------------------------------------------------
    q= "DELETE FROM ATTENDENCE_INCTURE.WEEK_WISE_ACT_PRED "                 
    cursor.execute(q)
    q= "DELETE FROM ATTENDENCE_INCTURE.WEEK_LEAVE_PRED "                 
    cursor.execute(q)
    q= "DELETE FROM ATTENDENCE_INCTURE.DAY_WISE_PRED "                 
    cursor.execute(q)
    q= "DELETE FROM ATTENDENCE_INCTURE.DAY_WISE_PRED_ACT"                 
    cursor.execute(q)
    q= "DELETE FROM ATTENDENCE_INCTURE.PROJ1_SUMMARY"                 
    cursor.execute(q)
    try:
        cursor.execute("COMMIT")
    except Exception as e:
        print( e ) 
    #----------------------------------------------------------
    df = day_working_hours_actual_pred
    for index,row in df.iterrows():
        id = row['EmpID']
        date = str(row['Date'])[0:10]
        p_hours = row['Predicted Working Hours']
        a_hours =row['Actual_Working_Hours']
        q= "INSERT INTO ATTENDENCE_INCTURE.DAY_WISE_PRED_ACT VALUES('"+ str(id) + "','" + str(date)+"'," + str(p_hours)+" ," + str(a_hours)  + ")"                    
        #print(q)
        cursor.execute(q)
    #----------------------------------------------------------
    df  = week_leave_pred
    for index,row in df.iterrows():
        id = row['EmpID']
        date = str(row['Date'])[0:10]
        p_hours = row['Predicted Leave']   
        q= "INSERT INTO ATTENDENCE_INCTURE.WEEK_LEAVE_PRED VALUES('"+ str(id) + "','" + str(date)+"'," + str(p_hours)+")"
        #print(q)
        cursor.execute(q)
    #----------------------------------------------------------
    df = day_working_hours_pred
    for index,row in df.iterrows():
        id = row['EmpID']
        date = str(row['Date'])[0:10]
        p_hours = row['Predicted Working Hours']
        q= "INSERT INTO ATTENDENCE_INCTURE.DAY_WISE_PRED VALUES('"+ str(id) + "','" + str(date)+"'," + str(p_hours)+")"                
        #print(q)
        cursor.execute(q)
    #----------------------------------------------------------
    df = week_working_hours_pred
    for index,row in df.iterrows():
        id = row['EmpID']
        date = str(row['Date'])[0:10]
        p_hours = row['Predicted Working Hours']
        q= "INSERT INTO ATTENDENCE_INCTURE.WEEK_WISE_ACT_PRED VALUES('"+ str(id) + "','" + str(date)+"'," + str(p_hours)+ ")"                    
        #print(q)
        cursor.execute(q)
    #----------------------------------------------------------
    df = Project_Prediction_summary
    df.fillna(0,inplace= True)
    for index,row in df.iterrows():
        id = row['EmpID']
        total_proj_hours = row['Total Project Duration(days)']
        delay = row['Delay In Project(days)']
        reason1 = row['Reason1']
        reason2 = row['Reason2']
        #p_hours = row['Predicted Working Hours']
        q= "INSERT INTO ATTENDENCE_INCTURE.PROJ1_SUMMARY VALUES ('"+  str(total_proj_hours) + "','" + str(int(delay)) +"','" + str(id) + "','" + str(reason1) + "','" + str(reason2) + "'"  +")"                    
       # print(q)
        cursor.execute(q)
    #----------------------------------------------------------
    try:
        cursor.execute("COMMIT")
    except Exception as e:
        print( e ) 



#----------------------------------------------------------------------------------




app=Flask(__name__,template_folder='templates')


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/postdata', methods=['POST'])
def post():  
    content = request.get_json()
    eid = content['ids'].split(",")
    duration = int (content['duration'])
    #master_program(duration ,eid)   
   
    return 'Success'

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000)

