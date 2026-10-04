"""Single full-stream GET. Called only by separately authorized execution."""
URL='https://data.carlanomaly.de/v1/carlanomaly-base-test.tar.gz'
EXPECTED_SIZE=91538225599

def open_one_shot(ledger):
    if ledger['http_requests']!=0:raise RuntimeError('SECOND_REQUEST_FORBIDDEN')
    import urllib.request
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):raise RuntimeError('REDIRECT_FORBIDDEN_SINGLE_REQUEST')
    opener=urllib.request.build_opener(NoRedirect())
    request=urllib.request.Request(URL,headers={'Accept-Encoding':'identity'},method='GET')
    if request.get_method()!='GET' or request.has_header('Range') or request.get_header('Accept-encoding')!='identity':raise RuntimeError('FULL_STREAM_REQUEST_POLICY_FAILED')
    ledger['http_requests']=1
    response=opener.open(request,timeout=120)
    try:
        if response.status!=200 or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('OFFICIAL_FULL_STREAM_HTTP_METADATA_MISMATCH')
        if int(response.headers.get('Content-Length','-1'))!=EXPECTED_SIZE:raise ValueError('OFFICIAL_COMPRESSED_LENGTH_MISMATCH')
    except BaseException:
        response.close();raise
    return response
