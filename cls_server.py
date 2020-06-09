# -*- coding: utf-8 -*-

import os
import json
import fasttext
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib import parse

HOST_NAME = '127.0.0.1'
PORT_NUMBER = 9210

root_path = os.path.dirname(__file__)
qa_cls_model = fasttext.load_model(os.path.join(root_path, 'model/qa_cls_model.bin'))
stop_words_file_path = 'data/stopwords.txt'
stop_words = open(stop_words_file_path, 'r').readlines()
stop_words = [word.strip() for word in stop_words]

class HttpHandler(BaseHTTPRequestHandler):
    """ Handler http request"""

    def do_GET(self):
        """ recieve http request
        GET type: http://127.0.0.1:9210/?question=刘翔身高多少
        """
        if '?' in self.path:
            try:
                # get query question
                query_string = parse.unquote(self.path.split('?', 1)[1])
                params = parse.parse_qs(query_string)
                url_question = params['question'][0]
                # get query tuple
                url_question = list(url_question)
                qa_cls_question = ' '.join([ts for ts in url_question if ts not in stop_words])
                qa_cls = qa_cls_model.predct(qa_cls_question)
                print(qa_cls)
                qa_cls = qa_cls[0][0]

                # response
                data = dict()
                data['response'] = qa_cls
                self.send_response(200)
                self.send_header('Content-type', 'text/json; charset=UTF-8')
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf8'))
            except IOError:
                self.send_error(404, 'Not Find Result')


if __name__ == '__main__':
    server = HTTPServer((HOST_NAME, PORT_NUMBER), HttpHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
