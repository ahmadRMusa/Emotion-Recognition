/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

#include <iostream>
#include <bitset>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Histograms the matrix
 * \author PW
 */
class LbpHistTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int maxTransitions READ get_maxTransitions WRITE set_maxTransitions RESET reset_maxTransitions STORED false)
    Q_PROPERTY(float min READ get_min WRITE set_min RESET reset_min STORED false)
    // if all LBPs with more transitions than maxTransitions should be put together into a single bin or be completely ignored.
    Q_PROPERTY(bool ignoreRest READ get_ignoreRest WRITE set_ignoreRest RESET reset_ignoreRest STORED false)

    BR_PROPERTY(int, maxTransitions, 2)
    BR_PROPERTY(float, min, 0)
    // if all LBPs with more transitions than maxTransitions should be put together into a single bin or be completely ignored.
    BR_PROPERTY(bool, ignoreRest, true)

    std::map <uchar, float> emptyHist;
    int binCount;

    void init()
    {
        binCount = countBins();
        initializeHist(emptyHist);
    }

    void project(const Template &src, Template &dst) const
    {
        //        const int dims = this->dims == -1 ? max - min : this->dims;

        std::vector<Mat> mv;
        split(src, mv);
        //Mat m(mv.size(), dims, CV_32FC1);

        //std::cout << "CV_8U: " << CV_8U << std::endl;

        for (size_t i=0; i<mv.size(); i++) { // each color channel
            Mat dstmat(1, binCount, CV_32F);
            int channels[] = {0};
            //            int histSize[] = {dims};
            //            float range[] = {min, max};
            //            const float* ranges[] = {range};
            std::map <uchar, float> hist(emptyHist);
            //std::cout << "hist columns:" << std::endl;
            for(std::map <uchar, float>::const_iterator it = hist.begin(); it != hist.end(); ++it)
            {
                std::stringstream msg5;
                msg5 << (int)(it->first) << ": " << (int)(it->second) << std::endl;
                //                std::cout << msg5.str();
            }
            Mat channel = mv[i];
            // calcHist requires F or U, might as well convert just in case
            //if (mv[i].depth() != CV_8U && mv[i].depth() != CV_32F)
            //  mv[i].convertTo(chan, CV_32F);
            //calcHist(&chan, 1, channels, Mat(), hist, 1, histSize, ranges);
            //memcpy(m.ptr(i), hist.ptr(), dims * sizeof(float));
            //std::cout << "image = "<< std::endl << mv[i] << std::endl << std::endl;
            // std::cout << mv[i].depth() << std::endl;
            for(int row = 0; row < channel.rows; ++row) {
                uchar* p = channel.ptr(row);
                for(int col = 0; col < channel.cols; ++col) {
                    *p++; //points to each pixel value in turn assuming a CV_8UC1 greyscale image
                    //                    std::stringstream msg1;
                    //                    msg1 << "p: '" << *p <<"' p(int): '" << (int)*p << "'eol" << std::endl;
                    //                    std::cout << msg1.str();
                    //                    if(hasMaxTransitions(*p)){

                    // if p exists in map. Does not exist if ignoreRest ist true and p is the lbp for the rest
                    if(hist.find(*p) != hist.end()){
                        hist[*p]++;
                    }

                    //                    std::cout <<std::endl <<std::endl <<std::endl <<std::endl;
                }
            }
            //std::cout << "Results:" << std::endl;
            int j = 0;
            for(std::map <uchar, float>::const_iterator it = hist.begin(); it != hist.end(); ++it)
            {
                std::stringstream msg3;
                msg3 << (it->first) << ": " << (it->second) << std::endl;
                //std::cout << msg3.str();
                dstmat.at<float>(0,j) = it->second;
                j++;
            }
//            std::cout << "result = "<< std::endl << dstmat << std::endl << std::endl;

            dst += dstmat;
        }

        //dst += m;
    }
    void initializeHist(std::map <uchar, float> &hist) const {

        //        uchar i = -1;
        //        do{
        //            i++;

        //            if(hasMaxTransitions(i)){
        //                hist[i] = 0;
        //            }
        //        } while(i != 255);

        uchar i = -1;
        do{
            i++;
            hist[i] = 0.0;

        } while(i != (binCount-1));
    }

    // Calculates how much bins are necessary to cover all bit combinations within the maximum number of transitions.
    int countBins() {
        int bins = 0;
        uchar i = -1;
        do{
            i++;

            if(hasMaxTransitions(i)){
                bins++;
            }
        } while(i != 255);
        if(!ignoreRest){ // add additional bin for LBPs which violate the maxtransitions
            bins++;
        }
        return bins;
    }

    bool hasMaxTransitions(uchar c) const {
        std::string binary = std::bitset<8>(c).to_string(); //to binary
        //        std::cout << binary << std::endl;
        //for(int i = 0; i<8; i++){

        //}
        //        for(char& c : binary) {
        //            std::cout << c << std::endl;

        //        }
        //        for(std::string::iterator it = binary.begin(); it != binary.end(); ++it) {
        //            std::cout << *it << std::endl;
        //        }
        char previousChar = '\0'; // used as null
        char firstChar = '\0';
        int transitionCounter = 0;
        for(std::string::size_type i = 0; i < binary.size(); ++i) { // iterate over chars
            char c = binary[i];
            if(previousChar != c && previousChar != '\0'){
                transitionCounter++;
            }

            if(i==0){
                firstChar = c;
            }else if(i==binary.size()-1){ // circular transition
                if(firstChar!=c){
                    transitionCounter++;
                }
            }
            previousChar = c;
        }
        if(transitionCounter>maxTransitions){
            return false;
        }else{
            return true;
        }
    }
};

BR_REGISTER(Transform, LbpHistTransform)

} // namespace br

#include "imgproc/lbphist.moc"
