clear
make install
cd D3Q27
# testExecutable
source cleanCase.sh
clear
testExecutable -GPU 0,1
fieldCalculate -calculationType containsNaN
fieldConvert -fileType vts
cd ../
