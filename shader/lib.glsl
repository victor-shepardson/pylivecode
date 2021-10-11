const float pi = 3.14159265359;

float sum(vec4 x){
  return dot(x,vec4(1.));
}
float sum(vec3 x){
  return dot(x,vec3(1.));
}
float sum(vec2 x){
  return dot(x,vec2(1.));
}
float mean(vec4 x){
  return sum(x)/4.;
}
float mean(vec3 x){
  return sum(x)/3.;
}
float mean(vec2 x){
  return sum(x)/2.;
}

float sigmoid(float x){
  return 1./(exp(-x)+1.);
}
vec2 sigmoid(vec2 x){
  return 1./(exp(-x)+1.);
}
vec3 sigmoid(vec3 x){
  return 1./(exp(-x)+1.);
}
vec4 sigmoid(vec4 x){
  return 1./(exp(-x)+1.);
}

vec4 samp(sampler2D t, vec2 px){
  return texture(t, px/vec2(textureSize(t, 0)));
}

vec4 blur4pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return (
    samp(t, px+d.xz)
    + samp(t, px+d.yz)
    + samp(t, px+d.zx)
    + samp(t, px+d.zy)
    )/4.;
}

vec4 blur5pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return (
    2.*samp(t, px)
    + samp(t, px+d.xz)
    + samp(t, px+d.yz)
    + samp(t, px+d.zx)
    + samp(t, px+d.zy)
    )/6.;
}

vec2 tex2latlon(vec2 uv){
  vec2 wh = vec2(
    // (uv.x - float(uv.x>0.5))*2.0,
    uv.x*2.0 - 1.0,
    uv.y*2.0 - 1.0
    );

  float lat = asin(wh.y)/pi;
  float lon = wh.x / cos(lat*pi);
  return vec2(lat, lon);
}

vec2 latlon2tex(vec2 latlon){
  float lat = (mod(latlon[0]+1.,2.)-1.);
  float lon = (mod(latlon[1]+1.,2.)-1.);
  return (vec2(
    lon * cos(pi*lat),
    sin(pi*lat)
    ) + 1.) / 2;
}

vec3 latlon2cart(vec2 latlon){
  float r = cos(pi*latlon.x);
  vec3 cart = vec3(
      r*cos(latlon.y*pi),
      r*sin(latlon.y*pi),
      sin(latlon.x*pi));
  return cart;
}

vec2 cart2latlon(vec3 cart){
  // float r = sqrt(dot(cart.xy, cart.xy));
  return vec2(
    atan(cart.y,cart.x)/pi,
    asin(cart.z)/pi
    );
}

vec2 wrap_latlon(vec2 latlon){
  latlon.x = mod(latlon.x+1., 2.) - 1.;
  if (abs(latlon.x) > 0.5){
    // if latitude exceeds (-0.5, 0.5), reflect it and add 1 to longitude
    latlon.x = (latlon.x>0?1.:-1.)-latlon.x;
    latlon.y += 1.;
  }
  latlon.y = mod(latlon.y+1., 2.)-1.;
  return latlon;
}

float bbsm = 1739.;
vec2 bbsopt(in vec2 a){
	return fract(a*a*(1./bbsm))*bbsm;
}
vec2 mod1024(in vec2 a){
	return fract(a*(1./1024.))*1024.;
}
vec4 hash(in vec2 pos){
	vec2 a0 = mod1024(pos*pi);
	vec2 a1 = bbsopt(a0);
	vec2 a2 = a1.yx + bbsopt(a1);
	vec2 a3 = a2.yx + bbsopt(a2);
	return fract((a2.xyxy + a3.xxyy + a1.xyyx)*(1./bbsm));
}
