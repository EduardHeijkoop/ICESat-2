void pnpoly(int nvert, int npts, float *vertx, float *verty, float *testx, float *testy, void *outdatav);

void pnpoly(int nvert, int npts, float *vertx, float *verty, float *testx, float *testy, void *outdatav)
{
    int i, j, k = 0;
    int c;
    int *outdata = (int *) outdatav;
    for (k=0; k<npts; ++k)
    {
        c = 0;
        for (i = 0, j = nvert-1; i < nvert; j = i++) {
            if ( ((verty[i]>testy[k]) != (verty[j]>testy[k])) &&
            (testx[k] < (vertx[j]-vertx[i]) * (testy[k]-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
            c = !c;
        }
        outdata[k] = c;
    }
}
