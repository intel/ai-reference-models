import requests
from requests.auth import HTTPBasicAuth
import urllib3
import os
from OneBOMGlobals import *

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)  # Disable certificate warnings


class OneBOMRequestManager():
  _token = None
  _client_secret = None
  _client_id = None
  CLIENT_ID_ENV_VAR = "CLIENT_ID"
  CLIENT_SECRET_ENV_VAR = "CLIENT_SECRET"
  TOKEN_API_ENDPOINT = "https://apis-internal.intel.com/v1/auth/token"
  VARIANT_DETAILS_ENDPOINT = "https://apis-internal.intel.com/item/v1/software-variant-details"
  RELEASE_DETAILS_ENDPOINT = "https://apis-internal.intel.com/item/v1/software-release-details"
  PROXIES = {
        "https": "http://proxy-dmz.intel.com:912/"
  }
  
  def __init__(self, use_token=True, ):
    super().__init__()
    
    self.get_client_id()
    self.get_client_secret()
    self._api_call_headers = None
    self._use_token = use_token
    self._session = requests.Session()
    self._session.trust_env = False
    self._session.verify = False  # Disable certificate verification
    self._token = None

    if use_token:
      self._token = self.set_token()

  def get_headers(self, use_token=True):
    # if we don't have a token, assume that is what we're trying to do
    headers = {}
    if not use_token:
      headers = {
      "Content-Type": "application/json", 
      "Cache-Control": "no-cache"
      }
    elif self._token:
      headers = {
        "Authorization": "Bearer " + self._token,
        "Cache-Control": "no-cache",
        "Content-Type": "application/json"
      }
    else:
      error("No token found.")
    return headers

  def set_token(self):
    if self._api_call_headers:
      return
    token_json = {
      "grant_type": "client_credentials",
      "client_id": f"{self.get_client_id()}",
      "client_secret": f"{self.get_client_secret()}"
    }
    if self._client_id is None or self._client_secret is None:
      raise ValueError("Client ID and Client Secret are required.")
    response = self.post(self.TOKEN_API_ENDPOINT, 
                         data=token_json,
                         allow_redirects=False,
                         verify=False
                        )
    if response.status_code == 200:
      tokens = json.loads(response.text)

      self._token = tokens.get("acccess_token")
      debug("SUCCESS: API token recieved.")
    else:
      raise ValueError(f"""Failed to get token. 
                        Status Code: {response.status_code}; 
                        Text: {response.text}; 
                        CLIENT_ID: {self._client_id};
                        CLIENT_SECRET: {self._client_secret}""")

  def get_client_id(self):
    self._client_id = os.getenv(OneBOMRequestManager.CLIENT_ID_ENV_VAR)
    if self._client_id is None:
      raise ValueError("Client ID required.")

  def get_client_secret(self):
    self._client_secret = os.getenv(OneBOMRequestManager.CLIENT_SECRET_ENV_VAR)
    if self._client_secret is None:
      raise ValueError("Client Secret required.")

  def get_proxies(self):
    return self.PROXIES

  def get(self, url, **kwargs):
    return self._session.get(url, proxies=self.get_proxies(), 
                            headers=self.get_headers(), **kwargs)

  def post(self, url, **kwargs):
    return self._session.post(url, proxies=self.get_proxies(), 
                        headers=self.get_headers(), 
                        **kwargs)

  def put(self, url, **kwargs):
    return self._session.put(url, proxies=self.get_proxies(), 
                        headers=self.get_headers(), 
                        **kwargs)

  def delete(self, url, **kwargs):
    return self._session.delete(url, proxies=self.get_proxies(), **kwargs)
  
class OneBOMSpeedRequestManager(OneBOMRequestManager):
  ITEM_ENDPOINT = "https://speed.intel.com/5/itembom.api/designitem/v1/itemCrud"
  BOM_ENDPOINT = "https://speed.intel.com/5/itembom.api/DesignItem/v1/BOMCrud"
  def __init__(self):
    super().__init__(False)

  def get_proxies(self):
    return {}

  def get_headers(self):
    return super().get_headers(False)
  
  def post(self, url, **kwargs):
    return self._session.post(url, proxies=self.get_proxies(), 
                            auth=HTTPBasicAuth(self._client_id, self._client_secret),
                            headers=self.get_headers(), 
                            **kwargs)

  def get(self, url, **kwargs):
    return self._session.get(url, proxies=self.get_proxies(), 
                             auth=HTTPBasicAuth(self._client_id, self._client_secret),
                             headers=self.get_headers(),
                             **kwargs)

  def put(self, url, **kwargs):
    return self._session.put(url, proxies=self.get_proxies(), 
                            headers=self.get_headers(), 
                            auth=HTTPBasicAuth(self._client_id, self._client_secret),
                            **kwargs)


