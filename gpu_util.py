import sys, os
import time
import subprocess
import getpass
import re
import urllib2
import cookielib

cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv"]
mem_free = 100
util_count = 0

username = raw_input("Enter way2sms username: ")
passwd = getpass.getpass()
message = "GPU is free. Login quickly"
number = raw_input("Send updates to number: ")
message = "+".join(message.split(' '))

start = time.time()

def sendSMS():
	"Sms sending protocol"
	
	#Logging into the SMS Site
	url = 'http://site24.way2sms.com/Login1.action?'
	data = 'username='+username+'&password='+passwd+'&Submit=Sign+in'
 
	#For Cookies:
	cj = cookielib.CookieJar()
	proxy = urllib2.ProxyHandler({'http': '10.3.100.207:8080', 'https': '10.3.100.207:8080'}) #remove proxy handler if already embedded in system settings.
	opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj), proxy)
 
	# Adding Header detail: fools webiste of being a browser
	opener.addheaders = [('User-Agent','Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.120 Safari/537.36')]
 
	try:
		usock = opener.open(url, data)
	except IOError:
		print "Error while logging in."
		return
 
 
	jession_id = str(cj).split('~')[1].split(' ')[0]
	send_sms_url = 'http://site24.way2sms.com/smstoss.action?'
	send_sms_data = 'ssaction=ss&Token='+jession_id+'&mobile='+number+'&message='+message+'&msgLen=136'
	opener.addheaders = [('Referer', 'http://site25.way2sms.com/sendSMS?Token='+jession_id)]
 
	try:
		sms_sent_page = opener.open(send_sms_url,send_sms_data)
		print "SMS has been sent."
	except IOError:
		print "Error while sending message"
		return

while 1:
	##sendSMS()
	output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
	temp_free = re.findall('[0-9]+', output)
	
	if len(temp_free) >= 1:
		mem_free = int(temp_free[0])
		if mem_free >= 2200:
			util_count+=1
			lapse = time.time()
		else:
			util_count = 0
			start = time.time()
	else:
		continue
	
	print mem_free

	if util_count >= 3:
		if (lapse - start) <= 3600:
			sendSMS()
			util_count = 0
			start = time.time()
		else:
			util_count = 0
			start = time.time()
	
	time.sleep(600)
		
