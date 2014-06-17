BEGIN {
	   print "/* [wxMaxima batch file version 1] [ DO NOT EDIT BY HAND! ]*/";
	   print "/* [ Created with wxMaxima version 13.04.0 ] */";
	   print "";
	   a=0;
}

/\\begin{maxima}/ && a==0 {
	a=1;
	print "/* [wxMaxima: input   start ] */";
	next;
}

/maximaoutput|\\end{maxima}/ && a == 1 {
	print "/* [wxMaxima: input   end   ] */";
	print "";
	a=0;
}a

END {
	print "";
	print "/* Maxima can't load/batch files which end with a comment! */";
	print "\"Created with wxMaxima\"$";
}
