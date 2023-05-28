using OMEinsum
function to4(A)
    D = size(A,1)
    A4 = ein"abcd,cefg,hijb,jkle->ahikfldg"(A,A,A,A)
    reshape(A4,(D^2,D^2,D^2,D^2))
end