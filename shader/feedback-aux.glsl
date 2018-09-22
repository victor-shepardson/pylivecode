uniform vec2 size;
uniform sampler2D history_t0_b0;
// uniform sampler2D history_t1_b0;
uniform sampler2D filtered;
uniform sampler2D aux;

uniform float drag = 0.95;

float eps = 1e-5;

out vec4 fragColor;

mat2x4 centroid5pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  vec4 c = samp(t, px);
  vec4 e = samp(t, px+d.xz);
  vec4 w = samp(t, px+d.yz);
  vec4 n = samp(t, px+d.zx);
  vec4 s = samp(t, px+d.zy);
  vec4 sigma = c+e+w+n+s+eps;
  return mat2x4(
    (e - w)/sigma,
    (n - s)/sigma);
}

void main() {
  vec2 px0 = gl_FragCoord.xy;
  vec2 px = px0;
  vec2 uv = px/size;

  vec4 c0 = samp(history_t0_b0, px);
  vec4 w0 = samp(filtered, px);
  vec4 a0 = samp(aux, px);

  vec4 d0 = mix(c0, w0, 0.5);
  vec2 drift;
  drift = 2.*vec2(0.5*sin(2.*pi*d0.a), 0.5-d0.a);
  // drift = vec2(0.,-1.);
  // drift = vec2(0);

  mat2x4 cent;
  vec4 w = w0;
  vec4 c_acc = vec4(0.);
  vec2 delta = vec2(0.);
  const int n = 5;
  float nf = 0.;
  for(int i=0; i<n; i++){
    nf+=1.;
    w = samp(filtered, px);//*(0.5+w.gbra)*2.;// + w.gbar;
    // w -= mean(w);//*0.5;
    w /= length(w)+eps;
    cent = centroid5pt(history_t0_b0, px);
    // delta = 2.*delta + sin(pi*(w.rg-w.ba));
    delta = -1.*w*cent;//*pow(1.5, float(i));
    // delta /= length(delta)+eps+0.05;
    px += delta;//(0.01+length(delta));

    // w = w.gbar;
    c_acc += samp(history_t0_b0, px);
    // c_acc = 1.*c_acc.gbar;
    // if(sin(2*pi*c_acc.a) > 1.-nf*2./float(n)) break;
    // if(w.a > 1. - nf/float(n)) break;
    // if(length(c_acc-mean(c_acc)) > sqrt(2.)*nf/float(n)) break;
    // vec4 cl = samp(history_t0_b0, px);
    // if(1.-cos(length(cl-mean(cl))) > 2.-2.*nf/float(n)) break;
  }
  // c_acc = 0.5+0.5*cos(sqrt(c_acc));
  // c_acc = sin(pi*c_acc/(abs(c_acc.a)+1.));
  // c_acc /= c_acc.a+(1.-w0.a)+eps;
  c_acc /= c_acc.a+0.25;
  // c_acc /= nf*1.5*sigmoid(4.*dot(w0-mean(w0), c_acc-mean(c_acc)))+eps;
  // c_acc /= mix(c_acc.g, c_acc.b, 1.-c_acc.a)+eps;
  // c_acc = exp(1.5*c_acc);
  // c_acc /= length(c_acc) + 1e-5;
  // c_acc /= sum(c_acc);
  // c_acc /= nf;
  // c_acc /= (length(c_acc) + nf)/2;

  vec4 c_sv = 0.05*cos(-2.*pi*(c_acc+uv.xxyy*vec4(1.,1.,1.,1.)+vec4(0.,.25,0.,.25)));
  // vec4 c_sv = 0.05*cos(-2.*pi*uv.xxxy*vec4(1.,2.,3.,1.));
  // c_sv += 0.1*w0;
  // c_sv += 0.05*sin(-2.*pi*samp(filtered, px+0.*(w0.rg-w0.ba)));

  px += drift;

  vec4 c1 = blur5pt(history_t0_b0, px);
  // vec4 w1 = samp(history_t1_b0, px);
  vec4 a1 = samp(aux, px);

  vec4 c2 = c_acc.gbar;//mix(c_acc.gbar, 1.-c_acc.barg, a0);//c_acc.argb;//c_acc.barg;

  vec4 sat = c2 - mean(c2.rgb);//(c2.r+c2.g+c2.b+c2.a)/4.;
  // sat /= length(sat)+eps;
  vec4 change = (c2-w0)/2.;
  // change /= length(change)+eps;
  vec4 sharp = samp(history_t0_b0, px)-c1;
  // sharp /= length(sharp)+eps;

  vec3 scsw = vec3(1., 1., -1.);
  vec4 scs = (sat*scsw.x + change*scsw.y + sharp*scsw.z);

  c2 += 0.1 * scs / (length(scs)+eps);

  // c2 += sat*(0.5-w.a)*0.1 + change*0.05;

  // c2 = fract(c2);
  // c2 = c_sv + c2;
  c2 = fract(c_sv + c2);
  // c2 = fract(c2*(c2-0.25+a0)*1.333);
  // c2 = fract(c2*(c2-0.25+a0)*1.333-c_sv/4.);
  // c2 = .5-.5*cos(2.*pi*c2);
  // c2 = sin(pi*c2);

  // c2 = 2.*c2-1. + c_sv;
  // float lc2 = length(c2);
  // if(lc2>1.) c2 *= -fract(lc2)/lc2;
  // c2 = 0.5*(c2+1.);

  // c2 = 0.5-0.5*cos(2.*pi*(c2+a1));

  vec4 c = mix(c2, c1, drag);

  // c = max(vec4(0.), min(vec4(1.), c-0.05*(a1-0.0)));
  c = mix(c, 1.0-c.argb, a0);


  // // vec4 h0 = c0;
  // // vec4 h1 = samp(history_t1_b0, px0);
  // vec4 h0 = c1;
  // // vec4 h0 = mix(c1, c0, drag);
  // vec4 h1 = samp(history_t1_b0, px + (px - px0));
  // // vec4 h1 = mix(
  // //   samp(history_t1_b0, px0 - (px - px0)),
  // //   samp(history_t1_b0, px0),
  // //   drag);
  // vec4 dh = h0 - h1;
  // vec4 dc = c - h0;
  // vec4 dh_hat = dh/(length(dh)+eps);
  // vec4 dc_hat = dc/(length(dc)+eps);
  // vec4 dc_pll = dh_hat * dot(dc, dh_hat);
  // vec4 dc_perp = dc - dc_pll;
  // // dc_pll = clamp(dc_pll, 0., 2.);//0.5*abs(dc_pll);
  // // dc_pll *= 0.5 + 0.5*sin(pi/2.*dot(dc_hat, dh_hat));
  // dc_pll *= clamp(1.+dot(dc_hat, dh_hat), 0., 1.);
  // dc_perp = dc_perp;//*(1.-drag);
  // dc = dc_pll + dc_perp;
  // c = h0 + dc;

  fragColor = c;//mix(c, c0, drag);
}
