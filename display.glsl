uniform vec2 size;
uniform sampler2D color;

const float pi = 3.14159265359;

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;

  vec4 c = texture2D(color, fract(p*1.0));

  gl_FragColor = vec4(
    // c.bbb*(0.1+1.2*c.aaa)+0.05*cos(2.*pi*vec3((2.*c.r-c.g)/3.,(c.r-c.g)/2.,(2.*c.g-c.r)/3.)-pi),
    c.rgb,
    1.);
}
