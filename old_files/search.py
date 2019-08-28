#! /usr/bin/env python3.3	
#Search interface that gives Flickr and Youtube Results

#make the root window
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from ast import literal_eval
from subprocess import check_output
from os import chdir, getcwd

home = getcwd()

root = Tk()

#finds a directory
# dirname = filedialog.askdirectory()

#state variables
query = StringVar()
# Results will change, but must be stated as an empty string. 
yt_results= StringVar()
fl_results= StringVar()
list_yt_results = StringVar()
list_fl_results = StringVar()
list_results = StringVar()
searchmsg = StringVar()
num_yt_results = IntVar()
num_fl_results = IntVar()
num_results = IntVar()
# list_selection = StringVar()
yt_list_selection = StringVar()
fl_list_selection = StringVar()
list_selection = StringVar()
selectiones = StringVar()
maxResult = IntVar()
colored_rows = StringVar()
maxResult.set(10)
colors = {'Youtube': '#f0f0ff', 'Flickr': '#ffccff'}
titles = ['Youtube', 'Flickr']

#Define in-action functions; in this case, define the search function that will 
#open a new terminal and run the youtube search function I've already made in 
#python 2.7 that doesn't import well into python 3.

# def download(*args):
	
def Search(*args):
	from os import chdir
	from subprocess import check_output
	l_results = []
	# try:
	if ch_box_yt.state()[0] == 'selected':
		q = query.get()
		# chdir('/Users/socialmedia/Desktop/GUI/youtubeAnalytics-cmd-line-sample/')
		maxRes = maxResult.get()
		search = check_output(['/usr/bin/python2.7','ytSearch.py', q, str(maxRes)])
		res = search.decode("utf-8")
		yt_results.set(res)
		make_list_yt_results()
		searchmsg.set("Showing results for " + q)
		for i in range(0, num_yt_results.get(), 2):
			yt_lbox.itemconfigure(i, background=colors['Youtube']) 
	if ch_box_yt.state()[0] == 'deselected':
		yt_results.set('')
		make_list_yt_results
	if ch_box_fl.state()[0]== 'selected':
		from subprocess import check_output
		# from os import chdir
		q = query.get()
		# chdir('/Users/socialmedia/Desktop/GUI/')
		maxRes = maxResult.get()
		search = check_output(['/usr/bin/python2.7','Flickr Search.py', q, str(maxRes)])
		res = search.decode("utf-8")
		fl_results.set(res)
		make_list_fl_results()
		# searchmsg.set("Showing results for " + q)
		for i in range(0, num_fl_results.get() ,2):
			fl_lbox.itemconfigure(i, background=colors['Flickr'])
		res = literal_eval(res)
	if ch_box_fl.state()[0] == 'deselected':
		fl_results.set('')
		make_list_fl_results() 
		# for line in literal_eval(res):
			# 	l_results += [line]
			# list_results = tuple(l_results)
	# except:
	# 	pass

def make_list_yt_results(*args):
	r = literal_eval(yt_results.get())
	num_yt_results.set(len(r))
	l = []
	for n in range(len(r)):
		l+=["%02d" % (n+1)+'.\t'+r[n]['title']]
	l =tuple(l)
	list_yt_results.set(l)

def make_list_fl_results(*args):
	r = literal_eval(fl_results.get())
	num_fl_results.set(len(r))
	l = []
	# for line in r:
	# 	l+=[line['title']] 
	for n in range(len(r)):
		l+=["%02d" % (n+1)+'.\t'+r[n]['title']]
	l =tuple(l)
	list_fl_results.set(l)

def addSelection(*args):
	try:
		yt_old = literal_eval(yt_list_selection.get())
	except:
		yt_old = []
	try:
		fl_old = literal_eval(fl_list_selection.get())
	except:
		fl_old = []
	try:
		fl_new = fl_lbox.curselection()
	except:
		fl_new = []
	try:
		yt_new = yt_lbox.curselection()
	except:
		yt_new = []
	yt_selection = set(yt_old) | set(yt_new)
	fl_selection = set(fl_old) | set(fl_new)
	yt_list_selection.set(sorted(tuple(yt_selection), key=float))
	fl_list_selection.set(sorted(tuple(fl_selection), key=float))
	showSelection()

def removeSelection(*args):
	try:
		yt_old = list(literal_eval(yt_list_selection.get()))
	except:
		yt_old = []
	try:
		fl_old = list(literal_eval(fl_list_selection.get()))
	except:
		fl_old = []
	ys = len(yt_old)
	fs = len(fl_old)
	remove = selection_list.curselection()
	# print(remove)
	yt_indeces = []
	fl_indeces = []
	for r in remove:
		if int(r) == 0 or ys+1:
			pass
		if int(r) < ys+1:
			yt_indeces += [int(r) - 1]
		if int(r) > ys+1:
			fl_indeces += [int(r) - (ys+2)]
	yt_r = []
	fl_r = []
	for i in yt_indeces:
		yt_r+= yt_old[i]
	for i in fl_indeces:
		fl_r+= fl_old[i]
	yt_new = set(yt_old) - set(yt_r)
	fl_new = set(fl_old) - set(fl_r)
	yt_list_selection.set(sorted(tuple(yt_new), key=float))
	fl_list_selection.set(sorted(tuple(fl_new), key=float))
	showSelection()
	
def showSelection(*args):
	try:
		yt_index = list(literal_eval(yt_list_selection.get()))
	except:
		yt_index = []
	try:
		fl_index = list(literal_eval(fl_list_selection.get()))
	except:
		fl_index = []
	indeces = [{'index':yt_index, 'title': 'Youtube', 'select': [], 'res': literal_eval(list_yt_results.get())}, 
			   {'index':fl_index, 'title': "Flickr", 'select':fl_index, 'res': literal_eval(list_fl_results.get())}]
	selections = ()
	colored_rows = {}
	for i in indeces:
		select = ()
		for d in i['index']:
			select += tuple([i['res'][int(d)]])
		select = tuple([i['title']])+tuple(select)
		colored_rows[len(selections)]= i['title']
		selections += select
	selectiones.set(selections)
	for r in range(len(selections)):
		if r in colored_rows.keys():
			selection_list.itemconfigure(r, background=colors[colored_rows[r]])
		else:
			selection_list.itemconfigure(r, background='')

def clearSelection(*args):
	yt_list_selection.set(())
	fl_list_selection.set(())
	showSelection()

def saveAs(*args):
	yt_selection = literal_eval(yt_list_selection.get())
	fl_selection = literal_eval(fl_list_selection.get())
	dirname = filedialog.askdirectory()
	chdir(dirname)
	try:
		yt_res = literal_eval(yt_results.get())
	except:
		pass
	try:
		fl_res = literal_eval(fl_results.get())
	except:
		pass
	for y in yt_selection:
		try:
			id = yt_res[int(y)]['description']['videoId']
			downloadYt(id)
		except:
			pass
	fl_ids = []
	for f in fl_selection:
		id = fl_res[int(f)]['id']
		fl_ids += [id]
	downloadFl(fl_ids, dirname)
	chdir(home)
	# for f in fl_selection:
	# 	id = 
		
def downloadYt(id):
	url = 'https://www.youtube.com/watch?v='+str(id)
	try:
		search = check_output(['youtube-dl', url, '-f', '137+140'])
	except:
		try:
			search = check_output(['youtube-dl', url])
		except:
			pass

def downloadFl(ids, dirname):
	urls = []
	for id in ids:
		url = check_output(['/usr/bin/python2.7', home+'/Flickr_get_Original.py', id])
		url = url[:-1]
		url = url.decode('utf-8')
		urls += ['-O', url]
	search = ['/usr/bin/curl']+urls
	chdir(dirname)
	check_output(search)

def preview(*args):
	#try all listbox variables for none-empty values; if selection is a list, preview only the first of them
	try:
		yt_curse = yt_lbox.curselection()
		yt_res = literal_eval(yt_results.get())
		id = yt_res[int(yt_curse[0])]['description']['videoId']
		previewYT(id)
	except:
		pass
	try:
		fl_curse = fl_lbox.curselection()
		fl_res = literal_eval(fl_results.get())
		id = fl_res[int(fl_curse[0])]['id']
		# print(id)
		url = check_output(['/usr/bin/python2.7',home+'/Flickr_get_medium.py', id])
		previewFL(url)
	except:
		raise 
	# try:
	# 	ser_curse = ser_lbox.curselection()

def previewYT(id):
	url = 'https://www.youtube.com/watch?v='+str(id)
	search = check_output(['/usr/bin/open', url, '-a', 'vlc'])

def previewFL(url):
	search = check_output(['/usr/bin/open','-g', url, '-a', 'vlc'])

# def preview(*args):


# Create and grid the outer content frame
c = ttk.Frame(root, padding=(10,10,12,0))
c.grid(column=0, row=0, sticky=(N,W,S,E))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0,weight=1)

#Create all widgets
query_entry = ttk.Entry(c, textvariable=query, width=50)
yt_lbox = Listbox(c, listvariable=list_yt_results, height=11, selectmode='extended')
fl_lbox = Listbox(c, listvariable=list_fl_results, height=11, selectmode='extended')
yt_lbl = ttk.Label(c, text="Youtube Search Results:", justify='left')
fl_lbl = ttk.Label(c, text="Flickr Search Results:", justify='left')
search = ttk.Button(c, text='Search', command=Search, default='active')
status = ttk.Label(c, textvariable=searchmsg, justify='right')
max_results = Spinbox(c, from_=10, to=50, textvariable=maxResult, width = 5)


# Make a sub-frame for the options and results
options = ttk.Frame(c, padding=0)
# ch_box_yt = ttk.Checkbutton(options, text='Youtube', instate='selected');
# ch_box_fl = ttk.Checkbutton(options, text='Flickr', instate='selected');
ch_box_yt = ttk.Checkbutton(options, text='Youtube');
ch_box_fl = ttk.Checkbutton(options, text='Flickr');
ch_box_local = ttk.Checkbutton(options, text='Local Server (Not Available yet)', state='disabled');
selection_list = Listbox(options, listvariable = selectiones, height=10, selectmode='extended')
clear = ttk.Button(options, text= 'clear', command=clearSelection)
save_as = ttk.Button(options, text= 'save as', command =saveAs, state='active')
ch_box_yt.grid(sticky=W, columnspan=2)
ch_box_fl.grid(stick=W, columnspan=2)
ch_box_local.grid(sticky=W, columnspan=2)
selection_list.grid(sticky=(S, N, W, E), columnspan=2)
clear.grid(column=0, sticky=(W,E))
save_as.grid(column=1, row=4, sticky=(W,E))
options.grid_rowconfigure(3, weight=1)
options.grid_columnconfigure(1, weight=1)

# make add and remove buttons in a frame
addRemove = ttk.Frame(c, padding=0)
addSelect = ttk.Button(addRemove, text= '>>', command=addSelection)#, width=2)
removeSelect = ttk.Button(addRemove, text= '<<', command=removeSelection)#, width=2)
# moreResults = ttk.Button(addRemove, text= 'more\nresults', command=removeSelection, width=2)
prevSelect = ttk.Button(addRemove, text = 'view', command = preview);
addRemove.grid(column=3, row=2, rowspan = 3)
addSelect.grid()
removeSelect.grid()
prevSelect.grid()

# Grid all the widgets
query_entry.grid(column=0, row=0, sticky=(W,E))
yt_lbl.grid(column=0, row= 1, columnspan=2, sticky=(W,E))
fl_lbl.grid(column=0, row= 3, columnspan=2, sticky=(W,E))
yt_lbox.grid(column=0, row=2, columnspan=2, sticky=(N,S,W,E))
fl_lbox.grid(column=0, row=4, columnspan=2, sticky=(N,S,W,E))
options.grid(column=4, row=0, rowspan=5, sticky=(W,N,S,E))
search.grid(column=1, row=0, columnspan=2, sticky=(W,E))
status.grid(column=0, row=5, columnspan=3, sticky=(S,W,E))
max_results.grid(column=1, row=5, sticky=(S,W,E))
c.grid_columnconfigure(0, weight=5)
c.grid_columnconfigure(4, weight=4)
c.grid_rowconfigure(2, weight=1)
c.grid_rowconfigure(4, weight=1)


yt_s = ttk.Scrollbar(c, orient=VERTICAL, command=yt_lbox.yview)
yt_lbox.configure(yscrollcommand=yt_s.set)
yt_s.grid(column = 2, row = 2, sticky=(W,N,S))
fl_s = ttk.Scrollbar(c, orient=VERTICAL, command=fl_lbox.yview)
fl_lbox.configure(yscrollcommand=fl_s.set)
fl_s.grid(column = 2, row = 4, sticky=(W,N,S))

root.bind('<Return>', Search)
yt_lbox.bind('<space>', preview)
fl_lbox.bind('<space>', preview)
yt_lbox.bind('<space>', preview)
fl_lbox.bind('<space>', preview)

ttk.Sizegrip().grid(column=3, row=6, sticky=(S,E))

query.set('')
yt_results.set('')
fl_results.set('')
num_yt_results.set(10)
num_fl_results.set(10)
yt_lbox.selection_set(0)
fl_lbox.selection_set(0)


root.mainloop()

