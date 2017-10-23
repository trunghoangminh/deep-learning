installPackagePip(){
	echo "======================== Starting install $1 via pip ========================="
	sudo pip install $1
	if [ $? == 0 ];
	then
		echo "Install $1 successful!"
	else
		echo "Install $1 fail!"
		exit 1
	fi
}

echo "We will install python in this machine"
echo "=============================================================================="
echo "========================== Starting install python 2 ======================"
sudo apt-get install python2.7
if [ $? == 0 ];
	then
	echo "Install python 2 successful!"
else
	echo "Install python 2 fail!"
	exit 1
fi
#######################################################################################
echo "================= Starting install python-pip and python-dev ===================="
sudo apt-get install python-pip python-dev
if [ $? == 0 ];
	then
	echo "Install python-pip and python-dev successful!"
else
	echo "Install python-pip and python-dev fail!"
	exit 1
fi
#######################################################################################
echo "============================= Starting install pip ========================="
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
sudo pip install --upgrade pip
if [ $? == 0 ];
	then
	echo "Install pip for python successful!"
	if [ -f "get-pip.py" ];
		then	
		rm get-pip.py
	fi
else
	echo "Install pip for python fail!"
	if [ -f "get-pip.py" ];
		then
		rm get-pip.py
	fi
	exit 1
fi
#######################################################################################
echo "============================= Starting update pip ========================="
pip install -U pip
if [ $? == 0 ];
	then
	echo "Update pip for python successful!"
else
	echo "update pip for python fail!"
	exit 1
fi
#######################################################################################
installPackagePip numpy

installPackagePip scipy

installPackagePip matplotlib

installPackagePip pandas

installPackagePip scikit-learn

installPackagePip keras

installPackagePip jupyter
#######################################################################################
echo "================= Starting install tensorflow for python ===================="
sudo pip install --upgrade \ https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl
if [ $? == 0 ];
	then
	echo "Install tensorflow for python successful!"
else
	echo "Install tensorflow for python fail!"
	exit 1
fi
#######################################################################################
echo "Installed all packages done!"
echo "======================================== Thank you (^-^)! ======================================="
