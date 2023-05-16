### Note
To tun this aplication the sentence encoder model in zip format has to be downloaded and stored in the 
root application folder. The model used for the application is 'universal-sentence-encoder_4.tar.gz', which has been downloaded from Tensorflow Hub.

Also, working and a valid OpenAI key has to be specified on line  on 14 on app.py file into the varable called 'openai_api_key'.

Finally, the application can be launched by running the command 

flask run -h localhost -p [PORT_NUMBER]