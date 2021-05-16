import numpy as np 
import pandas as pd
import streamlit as st
import h2o
from sklearn.model_selection import train_test_split
import s3fs
from h2o.estimators import H2ORandomForestEstimator
import collections
import base64
from transformers import pipeline

def main():
    st.title("Smart Water: Data Labeling with Active Learning and H2O.ai")
    st.markdown("## Step 1: Upload data from Local or S3")
    h2o.connect()
    data = None
    step_one = False
    @st.cache
    def load_Data_local(uploaded_files):
        if uploaded_files:
            for file in uploaded_files:
                file.seek(0)
            data_read = [pd.read_csv(file) for file in uploaded_files]
            data = pd.concat(data_read,ignore_index=True)
        else:
            return None
        return data
    @st.cache(suppress_st_warning=True)
    def load_Data_S3(value):
        if s3_path:
            h2o.init()
            data_h2o = h2o.import_file(s3_path)
            data = data_h2o.as_data_frame()
        else:
            return None, None
        return data_h2o, data

    upload_option = st.selectbox("Upload data from Local or S3:",('None','Local','S3'))
    if upload_option =='None':
        st.markdown("Please select upload option.")
    elif upload_option == 'Local':
        st.markdown("Only support CSV files")
        uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
        data = load_Data_local(uploaded_files)
        step_one = True
    elif upload_option == 'S3':
        s3_path = st.text_input("input s3 path")
        data_h2o, data = load_Data_S3(s3_path)
        step_one = True
        
    
    if st.checkbox("Preview the dataset"):
        st.markdown("Preview the data:")
        st.write(data.head())
    
    st.markdown("## Step 2: Define Tasks and Target Labels")
    step_two = False
    if step_one==True:
        task = st.selectbox("Select Task Type",('None',"Classification","NLP Topic Modeling","Image Annotation"))
        if task!='None':
            target = st.text_input("Input the Target Variable")
       
            if target not in set(data.columns):
                st.error("Selected Variable not in the dataset")
            else:
                labeled = data[~data[target].isna()].shape[0]
                unlabeled = data[data[target].isna()].shape[0]
                st.markdown(f"There are {labeled} labeled samples and {unlabeled} unlabeled samples.")
                step_two = True
        st.markdown("## Step 3: Prioritize Samples for Labeling with Active Learning")
        if step_two:
            labeled_df_h2o = h2o.H2OFrame(data[~data[target].isna()])
            unlabeled_df_h2o = h2o.H2OFrame(data[data[target].isna()])
            unlabeled_df_h2o = unlabeled_df_h2o.drop(target,axis=1)
            if task =='Classification':
                labeled_df_h2o[target] = labeled_df_h2o[target].asfactor()
            if task =='NLP Topic Modeling':
                
                text_col = st.text_input("Input the Column name of Text features")
                nlp_features = pipeline('feature-extraction')
                text_to_features = {}
                train_list = []
                unlabel_list = []
                labeled_df = []
                unlabel_df = []
                #labeled_text_col = list(labeled_df_h2o[:,text_col].as_data_frame())
                
                # for text,label in zip(labeled_df_h2o.as_data_frame()[text_col].tolist(),labeled_df_h2o.as_data_frame()[target].tolist()):
                #     text_to_features[text] = np.mean(nlp_features(text)[0],axis=0)
                #     labeled_df.append((text,text_to_features[text],label))

                for text in unlabeled_df_h2o.as_data_frame()[text_col].tolist():
                    text_to_features[text] = np.mean(nlp_features(text)[0],axis=0)
                    labeled_df.append((text,text_to_features[text]))

                # for i in range(data.shape[0]):
                #     text = data.loc[i,text_col]
                #     text_to_features[text] = np.mean(nlp_features(text)[0],axis=0)
                    
                #     if data.loc[i,target].isnan():
                #         unlabel_list.append(i)
                #         unlabel_df.append((text,text_to_features[text]))
                #     else:
                #         labeled_df.append((text,text_to_features[text],data.loc[i,target]))
                #         train_list.append(i)
                # data_vecs = pd.DataFrame(labeled_df,columns = ['text','embeddings','labels'])
                # unlabel_data_vecs = pd.DataFrame(unlabel_df,columns = ['text','embeddings'])

                # labeled_df_h2o = h2o.H2OFrame(data_vecs)
                # unlabeled_df_h2o = h2o.H2OFrame(unlabel_data_vecs)


            train, test = labeled_df_h2o.split_frame(ratios=[0.8])
            
            
            # Input parameters that are going to train
            if task =='Classification':
                training_columns = [i for i in data.columns if i!=target]
            if task =='NLP Topic Modeling':
                training_columns = 'embeddings'
            # Output parameter train against input parameters
            response_column = target
            
            
            # Define model
            model = H2ORandomForestEstimator()
            # Train model
            model.train(x=training_columns, y=response_column, training_frame = train,
                    validation_frame = test)

            # how many unlabel samples to be labeled per time
            def entropy(x):
                return -x*np.log(x+ 1e-10)
            
            predictions = model.predict(unlabeled_df_h2o)
            predictions = predictions.as_data_frame()
            @st.cache
            def get_samples(k):
                d = collections.defaultdict(int) # key: row id ;value:entropy
                for i in range(predictions.shape[0]):
                    for j in range(predictions.shape[1]-1):
                        prob = predictions.iloc[i,j+1]
                    d[i] += entropy(prob)
                idx_list = sorted(d, key=d.get,reverse = True)[:k]
                return idx_list

            k = int(st.number_input("Number of sample to get labeled per time(1-10, Integer)",step=1,value=5))
            
            if st.checkbox("Get Unlabeled Samples"):
                idx_list = get_samples(k)

                to_be_label = unlabeled_df_h2o.as_data_frame().iloc[idx_list,:].reset_index(drop=True)
                to_be_label.to_csv("temp_unlabeled.csv",index=None)
                unlabeled_df_h2o = unlabeled_df_h2o.drop(idx_list, axis=0)
                unlabeled_df_h2o.as_data_frame().to_csv("unlabeled_left.csv",index = None)
                st.write(to_be_label)
            annotated_data = pd.DataFrame(columns = data.columns)

            annotated_data_all = pd.DataFrame(columns = data.columns)

            if st.checkbox("Start labeling data (uncheck 'Get Unlabeled Samples' first)"):

                to_be_label = pd.read_csv("temp_unlabeled.csv")
                unlabeled_df_h2o = pd.read_csv("unlabeled_left.csv")
                st.write(to_be_label)
                for i in range(to_be_label.shape[0]):
                    st.write(to_be_label.iloc[i,:])
                labels = st.text_input("Input the labels seperated by ; (examples: [a;b;a;b;a] ")
                
                temp = []
                idx = data.columns.tolist().index(target)
                if labels:
                    for i,label in enumerate(labels.split(";")):
                        t=[0]*data.shape[1]
                        t[idx] = label.strip()
                        t[:idx] = to_be_label.iloc[i,:].tolist()[:idx]
                        t[idx+1:] = to_be_label.iloc[i,:].tolist()[idx:]
                        temp.append(t)
                    temp = pd.DataFrame(temp,columns = data.columns)
                    annotated_data = annotated_data.append(temp)
                    annotated_data.to_csv("temp_labeled.csv",index=None)
                    st.write(f"The Remaining unlabeled samples are {unlabeled_df_h2o.shape[0]}")
                    st.write(annotated_data)

                    
                    
                    data = data.append(annotated_data)
        
                #data,annotated_data,annotated_data_all = process_label(labels,data,annotated_data,annotated_data_all)
                def get_table_download_link(df,message):
                        
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                        href = f'<a href="data:file/csv;base64,{b64}">{message}</a>'
                        return href

                    
                    
                    # labeled + annotate + unlabeled
                df = pd.concat([labeled_df_h2o.as_data_frame(), unlabeled_df_h2o,annotated_data], axis=0)
                st.markdown(get_table_download_link(df,"Download data and label more"), unsafe_allow_html=True)
            
             
if __name__=='__main__':
    main()
