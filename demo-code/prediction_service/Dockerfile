FROM ubuntu:latest
MAINTAINER Linh H

RUN apt-get update \  
  && apt-get install -y python3.9 python3-pip \  
  && cd /usr/local/bin \  
  && ln -s /usr/bin/python3 python \  
  && pip3 install flask==2.0 pandas==1.4 mlflow==2.1 scikit-learn==1.0.2 statsmodels==0.13.2 gcsfs==2023.1.0 google-cloud-bigquery==3.0.1 db_dtypes
  
COPY web_endpoint.py model_trainer.py newacc_gcp_credential.json ./ 

ENTRYPOINT ["gunicorn","web_endpoint:app", "-b", "0.0.0.0:80"]