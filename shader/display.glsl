uniform vec2 size;
uniform sampler2D color;

out vec4 fragColor;

const float pi = 3.14159265359;

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;

  vec4 c = texture(color, p*1.0);

  float a = (texture(color, (px+vec2(0.,1.))/size) - c).a;
  float m = 8.*max(0.,a);

  fragColor = vec4(
    // c.bbb*(0.1+1.2*c.aaa)+0.05*cos(2.*pi*vec3((2.*c.r-c.g)/3.,(c.r-c.g)/2.,(2.*c.g-c.r)/3.)-pi),
    // c.rgb
    // mix(c.gbr, c.rgb, sqrt(m/(m+1.))),
    c.rgb*vec3(1., 0.33, 0.5) + c.rgb*sqrt(m/(m+1.))*vec3(0.33, 0.5, 0.5),
    1.);
}
