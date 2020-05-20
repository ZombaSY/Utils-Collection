from azure.storage.blob import BlobServiceClient
import os
import requests

# azure cloud parameter
account_name = 'digocloud'
azure_key = 'gajF2Kr5ybF1knSwG+B7vPUpMUFudoiPE583IIlOcxeWdX/rtnd5gescD1huOJ3+aI18fEAucDVxMdFnkGcX0g=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=digocloud;AccountKey=gajF2Kr5ybF1knSwG+B7vPUpMUFudoiPE583IIlOcxeWdX/rtnd5gescD1huOJ3+aI18fEAucDVxMdFnkGcX0g==;EndpointSuffix=core.windows.net'
local_path = os.getcwd() + '/temp'
local_file_name = '123456.txt'

# server parameter
DIGO_WEB = 'http://digo.ai:7777/digo-server'
DIGO_WEB = 'http://localhost:8080/'
api_key = 'DEFAULT4'
workspace_name = 'ZombaISSunyong'
project_name = 'project1'
experiment_name = 'experiment12353'

full_path_to_file = os.path.join(local_path, local_file_name)

# Write text to the file.
file = open(full_path_to_file, 'w')
file.write("Hello, World!")
file.close()


def upload_to_cloud(data, blob_service_client: BlobServiceClient, container_name: str):
    file_dir, file_name = os.path.split(data.name)
    save_name = project_name + os.path.sep + experiment_name + os.path.sep + file_name
    container_name = container_name.strip()     #

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=save_name)
    blob_client.upload_blob(data)


def save_blob_to_cloud(container_name):
    with open(full_path_to_file, 'rb') as file:
        # later: use 'isinstance' to check instance.
        conn: requests.Response = None

        file_size = os.path.getsize(file.name)

        if file_size == 0:
            raise Exception('No File Detected!!!')

        params = {'container_name': container_name, 'project_name': project_name,
                  'experiment_name': experiment_name, 'file_name': local_file_name, 'file_size': file_size}

        conn = requests.get(
            DIGO_WEB + "/CheckAzureCloud", params=params)

        json_data = conn.json()

        # Response from server. If code is SUCCESS.
        if json_data['code'] == 0:
            # Create the BlockBlockService that the system uses to call the Blob service for the storage account.

            blob_service_client = BlobServiceClient.from_connection_string(connection_string)

            upload_to_cloud(file, blob_service_client, container_name)

        elif json_data['code'] == -200:
            raise Exception('Not Enough Storage Left!!!')


def get_container_name():
    params = {'api_key': api_key, 'workspace_name': workspace_name}
    conn = requests.get(
        DIGO_WEB + "/GetContainerName", params=params
    )

    json_data = conn.json()

    return json_data['container_name']


def main():
    container_name = get_container_name()
    save_blob_to_cloud(container_name)


if __name__ == '__main__':
    main()
