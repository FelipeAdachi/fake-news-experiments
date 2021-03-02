from bravado.requests_client import RequestsClient
from bravado.client import SwaggerClient
import os
import pandas as pd
from io import StringIO
from definitions import lakefs_conf as conf

class lakefs_conn(object):
    def __init__(self):
        http_client = RequestsClient()
        http_client.set_basic_auth(conf['lakefs_url'], conf['lakefs_key_id'], conf['lakefs_secret_key'])
        client = SwaggerClient.from_url(
            'http://{}/swagger.json'.format(conf['lakefs_url']),
            http_client=http_client,
            config={"validate_swagger_spec": False})
        self.client = client
        # return client

    def get_csv(self,path_,ref_="master"):
        client = self.client
        x =client.objects.getObject(
        repository=conf['repository'],
        ref=ref_,
        path=path_,
        ).result()
        s=str(x,'utf-8')
        data = StringIO(s) 

        df=pd.read_csv(data)

        return df

    def create_branch_from_master(self,branch_name):
        client = self.client
        return client.branches.createBranch(
            repository=conf['repository'], branch={'name': branch_name, 'source': 'master'}).result()

    def upload_file(self,branch_name,input_path,dest_path):
        client = self.client
        with open(input_path, 'rb') as file_handle:
             res = client.objects.uploadObject(
                repository=conf['repository'],
                branch=branch_name,
                path=dest_path,
                content=file_handle
            ).result()
        return res

    def commit_to_branch(self,branch_name,message):
        client = self.client
        client.commits.commit(
            repository=conf['repository'],
            branch=branch_name,
            commit={
                'message': message,
                'metadata': {
                    'using': 'python_api'
                }
            }).result()