import os, glob, shutil, sys, re
from zipfile import ZipFile, ZIP_DEFLATED

EXINDEX = 4

simpleEmailRE = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"

def checkEmail(email):
	return re.fullmatch(simpleEmailRE, email) and email.endswith("unibas.ch")

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def cleanString(string):
	return string.strip().lower()

if __name__ == '__main__':
	print("Creating exercise .zip")
	pdfFiles = glob.glob("*.pdf")
	maxIndex = len(pdfFiles)-1
	for (i, file) in enumerate(pdfFiles):
		print(f' {i}) {file}')
	pdfIndex = int(input(f"Which .PDF file is your hand-in (enter integer index 0-{maxIndex})? >>> "))
	if pdfIndex > maxIndex or pdfIndex < 0:
		raise Exception("Out of range integer chosen!")
	elif not(os.path.exists("code") and os.path.isdir("code")):
		raise Exception("code directory does not exist!")
	else:
		expdforig = pdfFiles[pdfIndex]
		print(f" - File selected: {expdforig}")
		user1 = input("Group member 1 email: >>> ")
		user1 = cleanString(user1)
		if not checkEmail(user1):
			raise Exception(f"User 1 is not a valid unibas email: {user1}!")
		user2 = input("Group member 2 email (leave empty if only 1 person in the team): >>> ")
		user2 = cleanString(user2)
		if user2 and not checkEmail(user2):
			raise Exception(f"User 2 is not a valid unibas email: {user2}!")
		try:
			f = open("team", "w")
			f.write(f"{user1}\n{user2}")
			f.close()
			print("Team file created")
		except:
			raise("Could not create team file!")
		if user2:
			exname = f"PR-EX_{EXINDEX}_{user1}_{user2}"	
			print(f"Exercise .zip file being created for the group members '{user1}' and '{user2}'")
		else:
			exname = f"PR-EX_{EXINDEX}_{user1}"
			print(f"Exercise .zip file being created for the group '{user1}'")
		expdf = exname+'.pdf'
		exzip = exname+'.zip'
		try:
			os.mkdir(exname)
			print(f" - Temporary dir created: {exname}")
			shutil.copy("team", os.path.join(exname,"team"))
			print(f" - Copied team file")
			shutil.copy(expdforig, os.path.join(exname,expdf))
			print(f" - Copied exercise pdf file: {expdforig}")
			shutil.copytree("code", os.path.join(exname, "code"))
			print(" - Copied the code folder")
			zipf = ZipFile(exzip, 'w', ZIP_DEFLATED)
			zipdir(exname, zipf)
			zipf.close()
			print(f" - .zip file for upload created: {exzip}")
			os.remove("team")
			print(" - Temporary team file deleted")
			shutil.rmtree(exname)
			print(" - Temporary dir removed")
		except:
			raise(f"Error occured while creating submission .zip: {sys.exc_info()[0]}")